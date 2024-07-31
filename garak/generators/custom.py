from garak import _config
from garak.generators.rest import RestGenerator


class CustomRestGenerator(RestGenerator):
    """Custom REST generator that sends a POST request and retrieves the response with a subsequent GET request.

    Example configuration json;
    {
        "custom": {
            "CustomRestGenerator": {
                "name": "example service",
                "post_uri": "https://example.ai/llm",
                "post_headers": {
                    "X-Authorization": "$KEY"
                },
                "post_req_template_json_object": {
                    "text": "$INPUT"
                },
                "post_response_json": true,
                "post_response_json_field": "job_id",
                "get_uri": "https://example.ai/llm",
                "get_headers": {
                    "X-Authorization": "$KEY"
                },
                "get_req_template_json_object": {
                    "text": "$INPUT"
                },
                "post_response_json": true,
                "post_response_json_field": "text"
            }
        }
    }

    """

    DEFAULT_PARAMS = RestGenerator.DEFAULT_PARAMS | {
        "get_headers": RestGenerator.DEFAULT_PARAMS["headers"],
        "get_req_template": RestGenerator.DEFAULT_PARAMS["req_template"],
        "get_uri": "https://localhost",
        "post_headers": RestGenerator.DEFAULT_PARAMS["headers"],
        "post_response_json_field": "job_id",
        "post_req_template": RestGenerator.DEFAULT_PARAMS["req_template"],
        "post_uri": "https://localhost",
    }

    generator_family_name = "CustomREST"

    _supported_params = RestGenerator._supported_params + (
        "get_headers",
        "get_req_template",
        "get_req_template_json_object",
        "get_response_json",
        "get_response_json_field",
        "get_uri",
        "post_headers",
        "post_req_template",
        "post_req_template_json_object",
        "post_response_json",
        "post_response_json_field",
        "post_uri",
    )

    def request_config(self, method, config):
        result_config = {}
        result_config["method"] = method
        for attrib in RestGenerator._supported_params:
            method_param = f"{method.lower()}_{attrib}"
            if hasattr(self, method_param):
                result_config[attrib] = getattr(self, method_param)
            elif hasattr(self, attrib):
                result_config[attrib] = getattr(self, attrib)
        return result_config

    def __init__(self, uri=None, generations=10, config_root=_config):
        # Note: no call to super() will occur this generator wraps two RestGenerator objects
        self._load_config(config_root)

        module = self.__module__.split(".")[-1]
        klass = self.__class__.__name__

        gen_configs = (
            config_root.plugins.generators
            if hasattr(config_root, "plugins")
            else config_root["generators"]
        )

        sender_config = self.request_config("post", gen_configs[module][klass])
        sender_root = {
            "generators": {
                RestGenerator.__module__.split(".")[-1]: {
                    RestGenerator.__name__: sender_config
                }
            }
        }

        self.prompt_sender = RestGenerator(config_root=sender_root)

        response_config = self.request_config("get", gen_configs[module][klass])
        response_root = {
            "generators": {
                RestGenerator.__module__.split(".")[-1]: {
                    RestGenerator.__name__: response_config
                }
            }
        }

        self.response_retriever = RestGenerator(config_root=response_root)

    def _call_model(self, prompt: str, generations_this_call: int = 1):
        """Send a POST request and retrieve the response with a subsequent GET request."""

        job_id = self.prompt_sender._call_model(prompt, generations_this_call)
        self.response_retriever.uri = f"{self.get_uri}/{job_id}"
        return self.response_retriever._call_model(job_id, generations_this_call)


DEFAULT_CLASS = "CustomRestGenerator"
