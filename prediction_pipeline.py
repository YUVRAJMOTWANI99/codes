import json
import os
import re
from typing import Any

import pandas as pd
import torch
import yaml
from dotenv import load_dotenv
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chains import SequentialChain, TransformChain, LLMChain
from langchain.chains.base import Chain
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import FAISS
from torch.nn import functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    logging
)

from src.utils import json_parser

logging.set_verbosity_error()

# Load project settings
with open("setup.yaml", "r") as file:
    config = yaml.load(file, yaml.Loader)

# Load secret keys
load_dotenv()


class Pipeline:
    def __init__(self, pipeline: str, base_file_path: str):
        pipeline = pipeline.upper()
        self.pipeline = pipeline
        if self.pipeline == "P1":
            self.gpt_prediction_chain = self._get_gpt_prediction_chain(pipeline, base_file_path)
        elif self.pipeline == "P3":
            self.local_prediction_chain = self._get_local_prediction_chain(pipeline, base_file_path)
        elif self.pipeline == "P4":
            self.local_prediction_chain = self._get_local_prediction_chain(pipeline, base_file_path)
            self.evaluation_chain = self._get_gpt_eval_chain(pipeline, base_file_path)
        elif self.pipeline == "P5":
            self.local_prediction_chain = self._get_local_prediction_chain(pipeline, base_file_path)
            self.evaluation_chain = self._get_gpt_eval_chain(pipeline, base_file_path)
            self.detector_chain = self._get_gpt_detector_chain(pipeline, base_file_path)
            self.condense_chain = self._get_gpt_condense_question_chain(pipeline)
            self.oos_followup_question_chain = self._get_gpt_oos_followup_question_chain(pipeline, base_file_path)
            self.amb_followup_question_chain = self._get_gpt_amb_followup_question_chain(pipeline, base_file_path)

    def _get_gpt_prediction_chain(self, pipeline: str, base_file_path: str) -> Chain:
        """
        Define base prediction chain with GPT models
        """
        # Load embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name="all-mpnet-base-v2",
            cache_folder=f"{base_file_path}/embedding_models"
        )

        # Load example queries
        df = pd.read_csv(f"{base_file_path}/example_queries.csv")

        # Get examples for prediction task
        examples = df[["intent", "query"]].to_dict("records")
        example_template = """
        Example Query: {query}
        Example Category: {intent}
        """
        example_prompt = PromptTemplate(
            template=example_template,
            input_variables=["query", "intent"]
        )

        # Create example selector
        example_selector = SemanticSimilarityExampleSelector.from_examples(
            examples=examples,
            embeddings=embeddings,
            vectorstore_cls=FAISS,
            k=8
        )

        # Get class labels
        labels = df['intent'].unique().tolist()

        # Create final template
        prefix = """
You are a highly intelligent and accurate Multiclass Classification system. You take queries as input and classify that as one of the following appropriate Categories: [{labels}]

The output should only be the name of the category from the given list.

Examples:
        """

        suffix = """
END OF EXAMPLES

Query: {query}
Category:
        """
        intent_prediction_prompt = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=["query"],
            partial_variables={"labels": ", ".join(labels)},
            example_separator="\n"
        )

        # Prepare prediction chain
        openai_llm = AzureChatOpenAI(
            temperature=config[pipeline]["PREDICTION_PIPELINE"]["MODEL_TEMP"],
            max_tokens=50,
            openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
            deployment_name=config[pipeline]["PREDICTION_PIPELINE"]["DEPLOYMENT_NAME"],
            openai_api_version="2023-05-15",
            openai_api_base=config[pipeline]["PREDICTION_PIPELINE"]["AZURE_API_BASE"]
        )
        gpt_prediction_chain = LLMChain(
            llm=openai_llm,
            prompt=intent_prediction_prompt,
            output_key="intent",
            verbose=False
        )

        return gpt_prediction_chain

    def _get_local_prediction_chain(self, pipeline: str, base_file_path: str) -> Chain:
        """
        Function to initialize a local llm (bert like) prediction chain.
        This will load the model, wrap a langchain compatible API and use this model to create prediction chain
        """
        # Specify compute specs
        run_on = "cuda" if torch.cuda.is_available() else "cpu"

        # Load data
        df = pd.read_csv(f"{base_file_path}/preprocessed_data.csv")

        # Define label 2 id mappings
        labels = df["category"].unique().tolist()
        label2id = {row["category"]: row["category_id"] for idx, row in
                    df.drop_duplicates(["category", "category_id"]).iterrows()}
        id2label = {row["category_id"]: row["category"] for idx, row in
                    df.drop_duplicates(["category", "category_id"]).iterrows()}

        # Create tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            f"{base_file_path}/{config[pipeline]['FINETUNE']['MODEL_CHECKPOINT']}" + "_ft",
            add_prefix_space=True
        )

        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            f"{base_file_path}/{config[pipeline]['FINETUNE']['MODEL_CHECKPOINT']}" + "_ft",
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id
        ).to(run_on)

        # Create langchain wrapper around local llm
        class LLMWrap(LLM):
            """
            A wrapper around local_llm to be used in langchain
            """

            @property
            def _llm_type(self) -> str:
                return "localLLM"

            def _call(
                    self,
                    prompt: str,
                    stop: list[str] | None = None,
                    run_manager: CallbackManagerForLLMRun | None = None,
            ) -> str:
                if stop is not None:
                    raise ValueError("stop kwargs are not permitted.")

                # Inference
                inputs = tokenizer.encode(prompt, return_tensors="pt").to(
                    run_on)  # moving to mps for Mac (can alternatively do 'cpu')
                logits = model(inputs).logits
                sorted_predictions = torch.sort(logits, 1, descending=True).indices.tolist()[0]
                predicted_intent = id2label[sorted_predictions[0]]
                top_k_predictions = [id2label[i] for i in
                                     sorted_predictions[:config[pipeline]["PREDICTION_PIPELINE"]["TOP_K_PREDICTIONS"]]]
                probability_scores = F.softmax(logits, dim=-1).cpu().numpy()[0]
                confidence = float(max(probability_scores))
                to_return = {
                    "intent": predicted_intent,
                    "confidence": confidence,
                    "top_k_intents": top_k_predictions
                }

                # Return result
                return json.dumps(to_return)

        # JSON parser for adding extra parameter to the chain output
        def json_parser(inputs: dict) -> dict:
            result = json.loads(inputs["output"])
            inputs["intent"] = result["intent"]
            inputs["confidence"] = result["confidence"]
            inputs["top_k_intents"] = result["top_k_intents"]
            return inputs

        transform_chain = TransformChain(
            input_variables=["query", "output"], output_variables=["intent", "confidence", "top_k_intents"],
            transform=json_parser
        )

        # Prepare prediction chain
        local_prediction_chain_inter = LLMChain(
            llm=LLMWrap(),
            prompt=PromptTemplate(template="{query}", input_variables=["query"]),
            output_key="output"
        )
        local_prediction_chain = SequentialChain(
            chains=[local_prediction_chain_inter, transform_chain],
            input_variables=["query"],
            output_variables=["intent", "confidence", "top_k_intents"]
        )

        return local_prediction_chain

    def _get_gpt_eval_chain(self, pipeline: str, base_file_path: str) -> Chain:
        """
        Create evaluation chain.
        This chain is usually applied after prediction chain.
        """
        # Load data - get intent descriptions
        df_labels = pd.read_csv(f"{base_file_path}/intent_desc_3p_pov.csv")
        labels_with_desc = ""
        for _, row in df_labels.iterrows():
            labels_with_desc += f"label- {row['intent']} ; Meaning- {row['description']}\n"

        # Evaluation prompt
        intent_eval_prompt = PromptTemplate.from_template(
            """
Given a predicted intent for a user query, evaluate the accuracy of the predicted intent.
List of all available intents with their description is provided for your understanding.
If the description of the predicted intent does not map to user query, return NO. Otherwise, return YES.
If accuracy is NO, suggest ONE alternative intent that you think is most appropriate for the given user query.
Do not forget to suggest an alternate intent when accuracy is NO. Do not put extra text besides the suggested alternate intent.
Do not include explanation in your output; the explanations in examples are for your understanding only.

### List of Intents ###
{labels}## End of Intent List ##


### Examples ###
User Query: I want to add a new card
Predicted Intent: Card: report stolen or lost
Output: {{"accuracy": "NO",
"alternate_intent": "Card: add new"}}
Explanation- User wants to add a new card to their account. Card: add new intent is more appropriate in this case

User Query: Can I have more than one card?
Predicted Intent: Card: add new
Output: {{"accuracy": "YES",
"alternate_intent": ""}}
Explanation- Predicted intent correctly map to the user query according to the intent description

User Query: mere account mai kitne paise hai?
Predicted Intent: Spend Account: transfer funds
Output: {{"accuracy": "NO",
"alternate_intent": "Global: get balance"}}
Explanation- User query translates to what is my balance. Global: get balance intent is more appropriate in this case

User Query: I can't transfer money. Is my card working okay?
Predicted Intent: Transaction: report dispute incorrect
Output: {{"accuracy": "NO",
"alternate_intent": "Card: get info status"}}
Explanation- User is having problem transferring their funds and wants to check status of their card. Card: get info status intent is more appropriate in this case

User Query: where can i finds some atm nearby
Predicted Intent: User Account: get fee plan info
Output: {{"accuracy": "NO",
"alternate_intent": "Spending Account: find ATMs"}}
Explanation- User wants to know about the locations to withdraw their funds. Spending Account: find ATMs intent is more appropriate in this case

User Query: I want to report an incorrect transaction
Predicted Intent: Transaction: report dispute incorrect
Output: {{"accuracy": "YES",
"alternate_intent": ""}}
Explanation- Predicted intent maps correctly to the user query according to the intent description

User Query: Hello, I would like to view my historical transactions including all purchases, transfers, and deposits. Can you guide me on how to access this information?
Predicted Intent: Transaction: history
Output: {{"accuracy": "YES",
"alternate_intent": ""}}
Explanation- Predicted intent maps correctly to the user query according to the intent description

User Query: I need to talk to someone
Predicted Intent: Transaction: history
Output: {{"accuracy": "NO",
"alternate_intent": "User Account: get help contact customer service"}}
Explanation- User wants to talk to a customer service agent. User Account: get help contact customer service intent is more appropriate in this case.

User Query: Something is wrong with my historical transactions. I want to report that
Predicted Intent: Transaction: history
Output: {{"accuracy": "NO",
"alternate_intent": "Transaction: report dispute incorrect"}}
Explanation- User wants to dispute on of their old transaction. Transaction: report dispute incorrect intent is more appropriate in this case

## End of Examples ##

The output should be in JSON format as below:
{{
"accuracy": "predicted accuracy as YES or NO",
"alternate_intent": "One alternate intent if accuracy is NO"
}}

User Query: {query}
Predicted Intent: {intent}
Output:
            """,
            partial_variables={"labels": labels_with_desc}
        )

        # Prepare evaluation chain
        openai_llm = AzureChatOpenAI(
            temperature=config[pipeline]["PREDICTION_PIPELINE"]["EVAL_MODEL_TEMP"],
            max_tokens=400,
            openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
            deployment_name=config[pipeline]["PREDICTION_PIPELINE"]["EVAL_DEPLOYMENT_NAME"],
            openai_api_version="2023-05-15",
            openai_api_base=config[pipeline]["PREDICTION_PIPELINE"]["EVAL_AZURE_API_BASE"]
        )
        evaluation_chain_openai = LLMChain(
            llm=openai_llm,
            prompt=intent_eval_prompt,
            output_key="evaluation"
        )

        return evaluation_chain_openai

    def _get_gpt_detector_chain(self, pipeline: str, base_file_path: str) -> Chain:
        """
        Create a detection chain.
        This chain helps in detection of ambiguous user query.
        """
        # Load data - intent description
        df_labels = pd.read_csv(f"{base_file_path}/intent_desc_3p_pov.csv")
        labels_with_desc = ""
        for _, row in df_labels.iterrows():
            labels_with_desc += f"label- {row['intent']} ; Meaning- {row['description']}\n"

        # Detection prompt
        ambiguity_detection_prompt = PromptTemplate.from_template(
            """
You are an excellent classification engine. You pick up clever subtlety in inputs before concluding to a final prediction.
You have to classify a user query between three categories - [straightforward, ambiguous and out_of_scope].
User query provided in a different language should be classified as ambiguous only if the context is ambiguous - language barrier is not a factor for ambiguity.
Explanation is provided for your understanding in given examples. DO NOT include explanation in output.


### Description of each category ###
straightforward: User query accurately maps to an intent from the given list of intents.
ambiguous: User query is ambiguous and contains insufficient information. It can be mapped to more than one intent. User query being in different language should not be a factor of ambiguity - only use the context of user query. 
out_of_scope: User query does not map to any given intent. It is outside the scope of given list of intents.
### End of category list ###


Below is the list of intents. Put great focus on mapping between user query and given intents - it is vital for your prediction.
### List of Intents ###
{labels}## End of Intent List ##


### Examples ###
User Query: How much did i spend last month 
Output: {{"Predicted Category": "ambiguous"}}
Explanation: User query is ambiguous. It can be mapped to <Statement: get> as well as <Transaction: history>

User Query: Show me the way to those reward points!
Output: {{"Predicted Category": "ambiguous"}}
Explanation: User query is ambiguous. It can be mapped to <Rewards: opt in> as well as <Rewards: view offers>

User Query: where can i withdraw funds from my card
Output: {{"Predicted Category": "ambiguous"}}
Explanation: User query is ambiguous. It can be mapped to <Spend Account: get cash withdrawal and reload locations> as well as <Spend Account: find ATMs>

User Query: where is my card?
Output: {{"Predicted Category": "ambiguous"}}
Explanation: User query is ambiguous. It can be mapped to <Card: get shipping status where is> as well as <Card: report stolen or lost>

User Query: I want to watch a movie
Output: {{"Predicted Category": "out_of_scope"}}
Explanation: User query does not map to any given intent

User Query: quiero agregar una nueva tarjeta
Output: {{"Predicted Category": "straightforward"}}
Explanation: User query translates to 'I want to add a new card' which maps to only <Card: add new>. There is no scope for any other intent to be mapped to user query

User Query: I want to report a dispute transaction
Output: {{"Predicted Category": "straightforward"}}
Explanation: User query can be accurately mapped to <Transaction: report dispute incorrect>. There is no scope for any other intent to be mapped to user query

User Query: I want to apply for a car loan
Output: {{"Predicted Category": "out_of_scope"}}
Explanation: User query does not map to any given intent. There is no intent that corresponds to loan applications

User Query: I want to invest in stock market
Output: {{"Predicted Category": "out_of_scope"}}
Explanation: User query does not map to any given intent. There is no intent that corresponds to investing in stock market

User Query: mai paise bhejna chahta hun
Output: {{"Predicted Category": "straightforward"}}
Explanation: User query translates to 'I want to transfer funds' which maps to only <Spend Account: transfer funds>. There is no scope for any other intent to be mapped to user query

User Query: I want to edit my profile
Output: {{"Predicted Category": "ambiguous"}}
Explanation: User query is ambiguous. It can be mapped to <User Account: change email address>, <User Account: change mailing address>, <User Account: edit profile name> as well as <User Account: change password post login>

## End of Examples ##

The output should be in JSON format as below:
{{
"Predicted Category": "predicted category"
}}
The predicted category is one of [straightforward, ambiguous, out_of_scope]


User Query: {query}
Output:
            """,
            partial_variables={"labels": labels_with_desc}
        )

        # Prepare detection chain
        openai_llm = AzureChatOpenAI(
            temperature=config[pipeline]["PREDICTION_PIPELINE"]["DETECT_MODEL_TEMP"],
            max_tokens=100,
            openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
            deployment_name=config[pipeline]["PREDICTION_PIPELINE"]["DETECT_DEPLOYMENT_NAME"],
            openai_api_version="2023-05-15",
            openai_api_base=config[pipeline]["PREDICTION_PIPELINE"]["DETECT_AZURE_API_BASE"]
        )
        evaluation_chain_openai = LLMChain(
            llm=openai_llm,
            prompt=ambiguity_detection_prompt,
            output_key="detection"
        )

        return evaluation_chain_openai

    def _get_gpt_condense_question_chain(self, pipeline: str) -> Chain:
        """
        Create a question condenser chain.
        This chain consolidate two or more query into a standalone query without losing context.
        """
        # Condenser prompt
        query_condense_prompt = PromptTemplate.from_template(
            """
Given the following conversation and a follow up response, rephrase the follow up response from the user to be a standalone text, in English language.
Keep all the information/context present in chat history and followup response in the standalone response you will generate. Do not add extra phrases which were not in chat history and followup response.
Be precise and accurate.

Chat History:
User: {query}
AI: {followup_question}

Follow Up User Response: {query_2} 
Standalone response:
            """
        )

        # Prepare condenser chain
        openai_llm = AzureChatOpenAI(
            temperature=config[pipeline]["PREDICTION_PIPELINE"]["CONDENSE_MODEL_TEMP"],
            max_tokens=400,
            openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
            deployment_name=config[pipeline]["PREDICTION_PIPELINE"]["CONDENSE_DEPLOYMENT_NAME"],
            openai_api_version="2023-05-15",
            openai_api_base=config[pipeline]["PREDICTION_PIPELINE"]["CONDENSE_AZURE_API_BASE"]
        )
        condense_chain_openai = LLMChain(
            llm=openai_llm,
            prompt=query_condense_prompt,
            output_key="condensation",
            verbose=False
        )

        return condense_chain_openai

    def _get_gpt_oos_followup_question_chain(self, pipeline: str, base_file_path: str) -> Chain:
        """
        Create a followup question chain for out of scope inputs.
        This chain generate a context aware followup question when user input is out of scope.
        """
        # Load data - get intent descriptions
        df_labels = pd.read_csv(f"{base_file_path}/intent_desc_3p_pov.csv")
        labels_with_desc = ""
        for _, row in df_labels.iterrows():
            labels_with_desc += f"label- {row['intent']} ; Meaning- {row['description']}\n"

        # followup question generation prompt
        followup_question_prompt = PromptTemplate.from_template(
            """
Given an out of scope user query, generate a followup question, in English language.
User query is out of scope when it cannot be classified into any of the given intents.
The goal of this conversation is to find out the intent of the user. Therefore, followup question should politely inform user that their query is out of scope and consecutively provide alternate help. 
You have to phrase the followup question that should drive this conversation towards an intent.
Do not add extra information which is not given in the user query and intent descriptions. Do not produce a followup response that will divert this conversation towards an intent which is NOT given in the list of intents.
You have to be polite, helpful and respectful while generating the followup question.


### List of Intents ###
{labels}### End of Intent List ###

### Examples ###
User Query: Hello
Followup Question: Hi there, what can I do for you today?

User Query: I want to watch movies
Followup Question: Sorry, I cannot fulfill this request. I here to provide assistance with your banking needs.

User Query: uyafbisdlnkusgbiuldfng
Followup Question: Sorry, I cannot understand your query. Would you like to try again please?

User Query: I want to invest in stock market
Followup Question: Apologies, we can't directly assist with stock market investments, but we can help you check your balance before you start investing?

User Query: help me with my financial planning
Followup Question: Apologies, we can't directly assist with personal financial planning, but we can show you a spending tracker with insights into your past spending?

User Query: buy me a new laptop
Followup Question: Apologies, we can't help you purchase a laptop through this platform. But we can share your balance or suggest rewards programs to help you plan for this purchase?
### End of Examples ###

Remember, only suggest actions which are IN SCOPE of the given list of intents.


User Query: {query}
Followup Question:
            """,
            partial_variables={"labels": labels_with_desc}
        )

        # Prepare followup question chain
        openai_llm = AzureChatOpenAI(
            temperature=config[pipeline]["PREDICTION_PIPELINE"]["FOLLOWUP_GEN_MODEL_TEMP"],
            max_tokens=400,
            openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
            deployment_name=config[pipeline]["PREDICTION_PIPELINE"]["FOLLOWUP_GEN_DEPLOYMENT_NAME"],
            openai_api_version="2023-05-15",
            openai_api_base=config[pipeline]["PREDICTION_PIPELINE"]["FOLLOWUP_GEN_AZURE_API_BASE"]
        )
        followup_chain_openai = LLMChain(
            llm=openai_llm,
            prompt=followup_question_prompt,
            output_key="followup_question",
            verbose=False
        )

        return followup_chain_openai

    def _get_gpt_amb_followup_question_chain(self, pipeline: str, base_file_path: str) -> Chain:
        """
        Create a followup question chain for ambiguous inputs.
        This chain generate a context aware followup question when user input is ambiguous.
        """
        # Load data - get intent descriptions
        df_labels = pd.read_csv(f"{base_file_path}/intent_desc_3p_pov.csv")
        labels_with_desc = ""
        for _, row in df_labels.iterrows():
            labels_with_desc += f"label- {row['intent']} ; Meaning- {row['description']}\n"

        # followup question generation prompt
        followup_question_prompt = PromptTemplate.from_template(
            """
Given an ambiguous user query, generate a followup question, in English language.
User query is ambiguous when it can be classified in two or more intents.
The goal of this conversation is to find out the intent of the user. Therefore, followup question should ask the user to provide more context about their query so it can be classified into one of the given intents.
You have to phrase the followup question that should drive this conversation towards an intent.
Do not add extra information which is not given in the user query and intent descriptions. Do not produce a followup response that will divert this conversation towards an intent which is NOT given in the list of intents.
You have to be polite, helpful and respectful while generating the followup question.


### List of Intents ###
{labels}### End of Intent List ###

### Examples ###
User Query: where is my card
Followup  Question: Sure, do you want to know the shipping status of your card or you have lost it and want to report a missing card?

User Query: How much did i spend last month
Followup Question: Do you want to look at your statement or transaction history?

User Query: mereko cash nikalna hai apne card se
Followup Question: Would you like to use ATM or our withdrawal location network?

User Query: Show me the way to those reward points!
Followup Question: Do you want to "opt-in" for rewards points or you just want to see available rewards on your card?

User Query: Something is wrong with my card
Followup Question: I'm sorry to hear that. Can you please provide more details about what's wrong with your card?

User Query: My card is not working
Followup Question: I'm sorry to hear that. Can you please provide more details about what specifically is not working with your card?
### End of Examples ###

Remember, only suggest actions which are IN SCOPE of the given list of intents.


User Query: {query}
Followup Question:
            """,
            partial_variables={"labels": labels_with_desc}
        )

        # Prepare followup question chain
        openai_llm = AzureChatOpenAI(
            temperature=config[pipeline]["PREDICTION_PIPELINE"]["FOLLOWUP_GEN_MODEL_TEMP"],
            max_tokens=400,
            openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
            deployment_name=config[pipeline]["PREDICTION_PIPELINE"]["FOLLOWUP_GEN_DEPLOYMENT_NAME"],
            openai_api_version="2023-05-15",
            openai_api_base=config[pipeline]["PREDICTION_PIPELINE"]["FOLLOWUP_GEN_AZURE_API_BASE"]
        )
        followup_chain_openai = LLMChain(
            llm=openai_llm,
            prompt=followup_question_prompt,
            output_key="followup_question",
            verbose=False
        )

        return followup_chain_openai

    async def _eval_layer(self, query: str, intent: str) -> tuple[dict, str]:
        """
        Create evaluation layer using evaluation chain
        """
        # Call evaluation chain
        eval_result = await self.evaluation_chain.acall({"query": query, "intent": intent})
        eval_result = eval_result["evaluation"]

        # Parse llm output
        eval_result = json_parser(eval_result)

        # Extract suggested intent
        if eval_result["accuracy"] == "YES":
            return eval_result, ""

        new_intent = eval_result["alternate_intent"]
        return eval_result, new_intent

    async def _run_p1(self, query: str) -> dict[str: Any]:
        to_return = {}

        result = await self.gpt_prediction_chain.acall({"query": query})
        to_return["final_prediction"] = result["intent"].strip()

        return to_return

    async def _run_p3(self, query: str) -> dict[str: Any]:
        to_return = {}

        bert_result = self.local_prediction_chain(query)
        to_return["final_prediction"] = bert_result["intent"]
        to_return["final_confidence"] = bert_result["confidence"]
        to_return["final_top_k_intents"] = bert_result["top_k_intents"]

        return to_return

    async def _run_p4(self, query: str) -> dict[str: Any]:
        to_return = {"is_evaluated": False}

        # Bert only prediction
        bert_result = self.local_prediction_chain(query)
        to_return["final_confidence"] = bert_result["confidence"]
        to_return["final_top_k_intents"] = bert_result["top_k_intents"]

        # Check confidence
        if bert_result["confidence"] > config["P4"]["PREDICTION_PIPELINE"]["CONFIDENCE_THRESHOLD"]:
            to_return["final_prediction"] = bert_result["intent"]
            return to_return

        # Run evaluation chain - since it's a low confidence prediction from BERT
        eval_result, new_intent = await self._eval_layer(query, bert_result["intent"])
        to_return["is_evaluated"] = True
        if eval_result["accuracy"] == "YES":
            to_return["final_prediction"] = bert_result["intent"]
        else:
            to_return["final_prediction"] = new_intent
        return to_return

    async def _run_p5(self, query: str, query_2: str | None, followup_q: str | None) -> dict[str: Any]:
        to_return = {
            "final_prediction": "000x999",
            "is_evaluated": False,
            "followup_question": "00xx_no_followup_question_xx00",
            "final_input_to_bert": ""
        }

        while True:
            if not query_2:
                # Bert only prediction
                bert_result = self.local_prediction_chain(query)
                to_return["final_confidence"] = bert_result["confidence"]
                to_return["final_top_k_intents"] = bert_result["top_k_intents"]
                to_return["final_input_to_bert"] = query

                # Check confidence
                if bert_result["confidence"] > config["P5"]["PREDICTION_PIPELINE"]["CONFIDENCE_THRESHOLD"]:
                    to_return["final_prediction"] = bert_result["intent"]
                    return to_return

                # Call detector
                detector_result = await self.detector_chain.acall({"query": query})
                detector_result = json_parser(detector_result["detection"])

                # Check Ambiguity
                input_category = detector_result["Predicted Category"]
                if input_category == "straightforward":
                    # Run evaluation chain - since it's a low confidence prediction from BERT
                    eval_result, new_intent = await self._eval_layer(query, bert_result["intent"])
                    to_return["is_evaluated"] = True
                    if eval_result["accuracy"] == "YES":
                        to_return["final_prediction"] = bert_result["intent"]
                    else:
                        to_return["final_prediction"] = new_intent
                    return to_return
                elif input_category == "ambiguous":
                    # Generate followup question
                    followup_result = self.amb_followup_question_chain(query)
                    to_return["followup_question"] = followup_result["followup_question"]
                    return to_return
                elif input_category == "out_of_scope":
                    # Generate followup question
                    followup_result = self.oos_followup_question_chain(query)
                    to_return["followup_question"] = followup_result["followup_question"]
                    return to_return
                else:
                    msg = "Predicted category of input is not one of [straightforward, ambiguous, out_of_scope]. This prediction is performed by detector LLM."
                    raise RuntimeError(msg)
            else:
                # Condense Question
                condense_result = await self.condense_chain.acall(
                    {"query": query,
                     "followup_question": followup_q,
                     "query_2": query_2})

                # pass generated standalone question to BERT
                query = condense_result["condensation"]
                query_2 = ""

    async def run(self, query: str, query2: str | None = None, followup_q: str | None = None) -> dict[str: Any]:
        if self.pipeline == "P1":
            return await self._run_p1(query)
        if self.pipeline == "P3":
            return await self._run_p3(query)
        if self.pipeline == "P4":
            return await self._run_p4(query)
        if self.pipeline == "P5":
            return await self._run_p5(query, query2, followup_q)
