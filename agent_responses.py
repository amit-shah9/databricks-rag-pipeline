import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

from agent import grounded_agent_ask
import mlflow.models

class GroundedRAGResponsesAgent(ResponsesAgent):
    """Wraps grounded_agent_ask in MLflow's ResponsesAgent contract so Databricks
    can serve / evaluate / monitor it. This is a thin ADAPTER: unwrap the standard
    request -> call our existing agent -> wrap the answer in the standard response."""

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        # 1. UNWRAP: pull the latest user message text out of the standard request.
        #    request.input is a list of message items; take the last user one.
        user_text = ""
        for item in request.input:
            # items expose role/content; the user's question is the last user turn
            role = getattr(item, "role", None) or (item.get("role") if isinstance(item, dict) else None)
            if role == "user":
                content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
                user_text = content if isinstance(content, str) else str(content)
        if not user_text:
            user_text = str(request.input[-1])

        # 2. CALL our existing agent, unchanged.
        answer = grounded_agent_ask(user_text, verbose=False)

        # 3. WRAP: build a standard output item + response using MLflow helpers.
        output_item = self.create_text_output_item(text=answer, id="msg_1")
        return ResponsesAgentResponse(output=[output_item])
    
# Mark the servable agent instance for "models from code" logging.
AGENT = GroundedRAGResponsesAgent()
mlflow.models.set_model(AGENT)