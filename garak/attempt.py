"""Defines the Attempt class, which encapsulates a prompt with metadata and results"""

import json
import logging
from copy import deepcopy
from pathlib import Path
from types import GeneratorType
from typing import Any, List, Union
import uuid
from contextlib import contextmanager
import jinja2
from jinja2.ext import Extension
from jinja2.sandbox import ImmutableSandboxedEnvironment
from functools import lru_cache

(
    ATTEMPT_NEW,
    ATTEMPT_STARTED,
    ATTEMPT_COMPLETE,
) = range(3)

roles = {"system", "user", "assistant"}


class Turn:
    """Object to represent a single turn posed to or received from a generator
    Turns can be prompts, replies, system prompts. While many prompts are text,
    they may also be (or include) images, audio, files, or even a composition of
    these. The Turn object encapsulates this flexibility.
    `Turn` doesn't yet support multiple attachments of the same type.

    :param text: Text of the prompt/response
    :type text: str
    :param role: Role of the participant who issued the utterance Expected: ["system", "user", "assistant"]
    :type role: str
    :param data_path: Path to attachment
    :type data_path: Union[str, Path]
    :param data: Data to attach
    :type data: Any
    :param notes: Free form dictionary of notes for the turn
    :type notes: dict
    """

    @property
    def text(self) -> Union[None, str]:
        return self._text

    @text.setter
    def text(self, value: Union[None, str]) -> None:
        self._text = value

    @property
    def data(self):
        if self._data is not None:
            return self._data
        elif self._data is None and self.data_path is not None:
            self._load_data(self.data_path)
            return self._data
        else:
            raise ValueError(
                "Attempting to access `data` property of Turn but no data object or data_path was provided"
            )

    @data.setter
    def data(self, value: Any) -> None:
        if isinstance(value, str) or isinstance(value, Path):
            if self.data_path is not None and str(value) != str(self.data_path):
                logging.warning(
                    f"`Loading `data` property for Turn from {str(value)}, even though expected `data_path` for Turn is currently {str(self.data_path)}"
                )
            self._load_data(value)
        else:
            if self._data is not None:
                logging.warning(
                    f"Loading data for Turn even though `data` is already set."
                )

            self._data = value

    def __init__(
        self,
        text: Union[None, str] = None,
        role: Union[None, str] = None,
        data_path: Union[None, str] = None,
        data: Any = None,
        notes: dict = dict(),
    ):
        self._text = text
        self.role = role
        self.data_path = Path(data_path)
        if not self.data_path.is_file():
            raise ValueError(f"Provided `data_path` {data_path} is not a file.")
        self._data = data
        self.notes = notes

    def _load_data(self, data_path: Union[str, Path]):
        with open(data_path, "rb") as f:
            self._data = f.read()

    def to_dict(self) -> dict:
        parts = {"text": self._text}
        if self.role is not None:
            parts["role"] = self.role
        if self.data_path is not None:
            parts["data_path"] = str(self.data_path)
        if self._data is not None:
            parts["data"] = self._data
        if self.notes is not None:
            parts["notes"] = self.notes
        return parts

    @classmethod
    def from_dict(cls, message: dict):
        if not isinstance(message, dict):
            raise TypeError(
                f"Cannot create a `Turn`. Expected a dict but got {type(message)}"
            )
        # "content" is the key used by HF conversations, so let's support that formatting too.
        if "text" or "content" not in message.keys():
            raise AttributeError(
                "Cannot create a `Turn` from a dictionary without a `text` or `content` key."
            )
        elif "text" in message.keys():
            if isinstance(message["text"], str):
                text = message["text"]
            else:
                raise TypeError(
                    f"Turn `text` expects a str but got {type(message['text'])}"
                )
        else:
            if isinstance(message["content"], str):
                text = message["content"]
            else:
                raise TypeError(
                    f"Turn `text` expects a str but got {type(message['content'])}"
                )
        if "role" in message.keys():
            if isinstance(message["role"], str):
                role = message["role"]
            else:
                raise TypeError(
                    f"Turn `role` expects a str but got {type(message['role'])}"
                )
        else:
            role = None
        if "data" in message.keys():
            # No type checking here but maybe there should be.
            data = message["data"]
        else:
            data = None
        if "data_path" in message.keys():
            if isinstance(message["data_path"], str) or isinstance(
                message["data_path"], Path
            ):
                data_path = message["data_path"]
            else:
                raise TypeError(
                    f"Turn `data_path` expects a str or Path but got {type(message['data_path'])}"
                )
        else:
            data_path = None
        if "notes" in message.keys():
            if isinstance(message["notes"], dict):
                notes = message["notes"]
            else:
                raise TypeError(
                    f"Turn `notes` expects a dictionary but got {type(message['notes'])}"
                )
        else:
            notes = dict()
        return cls(text=text, role=role, data_path=data_path, data=data, notes=notes)

    def __str__(self):
        if self.data_path is None and self._data is None and self.role is None:
            return self._text
        else:
            parts = self.to_dict()
            return f"<Turn {repr(parts)}>"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Turn):
            return False
        match other:
            case _ if self.text != other.text:
                return False
            case _ if self.to_dict() != other.to_dict():
                return False
            case _:
                return True


class Conversation:
    """Class to maintain a sequence of Turn objects and, if relevant, apply conversation templates.
    :param messages: A list of dictionaries or Turns
    :type messages: list
    :param template: Jinja template for formatting conversations
    :type template: str
    :param notes: Free form dictionary of notes for the conversation
    :type notes: dict
    """

    def __init__(
        self,
        messages: list[Union[dict, Turn]] = None,
        template: str = None,
        notes: dict = {},
    ):
        if messages is None:
            self.messages = list()
        else:
            for message in messages:
                if isinstance(message, dict):
                    self.messages.append(Turn.from_dict(message))
                elif isinstance(message, Turn):
                    self.messages.append(message)
                else:
                    raise TypeError(
                        f"Conversations can only be constructed from Turn or dict objects but got {type(message)}"
                    )
        if len(messages) > 0:
            for message in messages:
                if message.role == "user":
                    self.initial_user_prompt = message.text
                    break
        self.template = template
        self.notes = notes

    def append(self, message: Union[Turn, dict]) -> None:
        if isinstance(message, Turn):
            self.messages.append(message)

        elif isinstance(message, dict):
            self.messages.append(Turn.from_dict(message))

        else:
            raise TypeError("`Conversation` objects can only contain `Turn`s")

    def apply_template(self) -> list[str]:
        if len(self.messages) < 1:
            logging.warning("Cannot apply a template to an empty Conversation")
            return ""
        if self.template is None:
            logging.warning(
                "Attempted to apply a chat template to Conversation but none is specified."
            )
            return [turn.text for turn in self.messages]
        compiled_template = self._compile_template()
        rendered_chats = list()
        for turn in self.messages:
            rendered_chat = compiled_template.render(messages=turn.text)
            rendered_chats.append(rendered_chat)
        return rendered_chats

    def clear_history(self):
        self.messages = list()

    def latest_message(self):
        if len(self.messages) > 0:
            return self.messages[-1]
        else:
            raise ValueError(
                "Attempted to return latest message from Conversation but message history is empty."
            )

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, key):
        return self.messages[key]

    def __iter__(self):
        yield from self.messages

    @lru_cache
    def _compile_template(self):
        # Largely borrowed and gently modified from HuggingFace's implementation of the same
        # https://github.com/huggingface/transformers/blob/main/src/transformers/utils/chat_template_utils.py#L365
        class AssistantTracker(Extension):
            # This extension is used to track the indices of assistant-generated tokens in the rendered chat
            tags = {"generation"}

            def __init__(self, environment: ImmutableSandboxedEnvironment):
                # The class is only initiated by jinja.
                super().__init__(environment)
                environment.extend(activate_tracker=self.activate_tracker)
                self._rendered_blocks = None
                self._generation_indices = None

            def parse(self, parser: jinja2.parser.Parser) -> jinja2.nodes.CallBlock:
                lineno = next(parser.stream).lineno
                body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
                return jinja2.nodes.CallBlock(
                    self.call_method("_generation_support"), [], [], body
                ).set_lineno(lineno)

            @jinja2.pass_eval_context
            def _generation_support(
                self, context: jinja2.nodes.EvalContext, caller: jinja2.runtime.Macro
            ) -> str:
                rv = caller()
                if self.is_active():
                    # Only track generation indices if the tracker is active
                    start_index = len("".join(self._rendered_blocks))
                    end_index = start_index + len(rv)
                    self._generation_indices.append((start_index, end_index))
                return rv

            def is_active(self) -> bool:
                return self._rendered_blocks or self._generation_indices

            @contextmanager
            def activate_tracker(
                self, rendered_blocks: List[int], generation_indices: List[int]
            ):
                try:
                    if self.is_active():
                        raise ValueError(
                            "AssistantTracker should not be reused before closed"
                        )
                    self._rendered_blocks = rendered_blocks
                    self._generation_indices = generation_indices

                    yield
                finally:
                    self._rendered_blocks = None
                    self._generation_indices = None

        def jinja_exception(msg):
            raise jinja2.exceptions.TemplateError(msg)

        def to_json(
            text, ensure_ascii=False, indent=None, separators=None, sort_keys=False
        ):
            return json.dumps(
                text,
                ensure_ascii=ensure_ascii,
                indent=indent,
                separators=separators,
                sort_keys=sort_keys,
            )

        jinja_env = ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=[AssistantTracker, jinja2.ext.loopcontrols],
        )
        jinja_env.filters["tojson"] = to_json
        jinja_env.globals["raise_exception"] = jinja_exception

        return jinja_env.from_string(self.template)


class Attempt:
    """A class defining objects that represent everything that constitutes a single attempt at evaluating an LLM.

    :param status: The status of this attempt; ``ATTEMPT_NEW``, ``ATTEMPT_STARTED``, or ``ATTEMPT_COMPLETE``
    :type status: int
    :param prompt: The processed prompt that will presented to the generator
    :type prompt: Turn
    :param probe_classname: Name of the probe class that originated this ``Attempt``
    :type probe_classname: str
    :param probe_params: Non-default parameters logged by the probe
    :type probe_params: dict, optional
    :param targets: A list of target strings to be searched for in generator responses to this attempt's prompt
    :type targets: List(str), optional
    :param outputs: The outputs from the generator in response to the prompt
    :type outputs: List(Turn)
    :param notes: A free-form dictionary of notes accompanying the attempt
    :type notes: dict
    :param detector_results: A dictionary of detector scores, keyed by detector name, where each value is a list of scores corresponding to each of the generator output strings in ``outputs``
    :type detector_results: dict
    :param goal: Free-text simple description of the goal of this attempt, set by the originating probe
    :type goal: str
    :param seq: Sequence number (starting 0) set in :meth:`garak.probes.base.Probe.probe`, to allow matching individual prompts with lists of answers/targets or other post-hoc ordering and keying
    :type seq: int
    :param messages: conversation turn histories; list of list of dicts have the format {"role": role, "content": text}, with actor being something like "system", "user", "assistant"
    :type messages: List(dict)
    :param lang: Language code for prompt as sent to the target
    :type lang: str, valid BCP47
    :param reverse_translation_outputs: The reverse translation of output based on the original language of the probe
    :param reverse_translation_outputs: List(str)

    Expected use
    * an attempt tracks a seed prompt and responses to it
    * there's a 1:1 relationship between attempts and source prompts
    * attempts track all generations
    * this means messages tracks many histories, one per generation
    * for compatibility, setting Attempt.prompt will set just one turn, and this is unpacked later
      when output is set; we don't know the # generations to expect until some output arrives
    * to keep alignment, generators need to return aligned lists of length #generations

    Patterns/expectations for Attempt access:
    .prompt - returns the first user prompt
    .outputs - returns the most recent model outputs
    .latest_prompts - returns a list of the latest user prompts

    Patterns/expectations for Attempt setting:
    .prompt - sets the first prompt, or fails if this has already been done
    .outputs - sets a new layer of model responses. silently handles expansion of prompt to multiple histories. prompt must be set
    .latest_prompts - adds a new set of user prompts


    """

    def __init__(
        self,
        status=ATTEMPT_NEW,
        prompt=None,
        probe_classname=None,
        probe_params=None,
        targets=None,
        notes=None,
        detector_results=None,
        goal=None,
        seq=-1,
        lang=None,  # language code for prompt as sent to the target
        reverse_translation_outputs=None,
    ) -> None:
        self.uuid = uuid.uuid4()
        if prompt is not None:
            self.prompt = Turn(text=prompt, role="user")
            self.messages = [Conversation([self.prompt])]
        else:
            self.prompt = prompt
            self.messages = list()

        self.status = status
        self.probe_classname = probe_classname
        self.probe_params = {} if probe_params is None else probe_params
        self.targets = [] if targets is None else targets
        self.notes = {} if notes is None else notes
        self.detector_results = {} if detector_results is None else detector_results
        self.goal = goal
        self.seq = seq
        self.lang = lang
        self.reverse_translation_outputs = (
            {} if reverse_translation_outputs is None else reverse_translation_outputs
        )

    def as_dict(self) -> dict:
        """Converts the attempt to a dictionary."""
        return {
            "entry_type": "attempt",
            "uuid": str(self.uuid),
            "seq": self.seq,
            "status": self.status,
            "probe_classname": self.probe_classname,
            "probe_params": self.probe_params,
            "targets": self.targets,
            "prompt": self.prompt.to_dict(),
            "outputs": [o.to_dict() for o in list(self.outputs)],
            "detector_results": {k: list(v) for k, v in self.detector_results.items()},
            "notes": self.notes,
            "goal": self.goal,
            "messages": [message.to_dict() for message in self.messages],
            "lang": self.lang,
            "reverse_translation_outputs": list(self.reverse_translation_outputs),
        }

    @property
    def prompt(self) -> Turn:
        if len(self.messages) == 0:  # nothing set
            return None
        else:
            try:
                return self.messages[0].initial_user_prompt
            except:
                raise ValueError(
                    "Message history of attempt uuid %s in unexpected state, sorry: "
                    % str(self.uuid)
                    + repr(self.messages)
                )

    @property
    def outputs(self):
        generated_outputs = list()
        if len(self.messages) and isinstance(self.messages[0], Turn):
            for conversation in self.messages:
                # work out last_output_turn that was assistant
                assistant_turns = [
                    idx for idx, val in enumerate(conversation) if val.role == "assistant"
                ]
                if not assistant_turns:
                    continue
                last_output_turn = max(assistant_turns)
                # return these (via list compr)
                generated_outputs.append(conversation[last_output_turn].text)
        return generated_outputs

    @property
    def latest_prompts(self):
        if len(self.messages) > 1:
            # work out last_output_turn that was user
            last_output_turn = max(
                [idx for idx, val in enumerate(self.messages) if val["role"] == "user"]
            )
            # return these (via list compr)
            return [self.messages[last_output_turn].text]
        else:
            return (
                self.prompt
            )  # returning a string instead of a list tips us off that generation count is not yet known

    @property
    def all_outputs(self):
        all_outputs = []
        if len(self.messages) > 0:
            for turn in self.messages:
                if turn.role == "assistant":
                    all_outputs.append(turn.text)
        return all_outputs

    @prompt.setter
    def prompt(self, value: str | Turn):
        if value is None:
            raise TypeError("'None' prompts are not valid")
        if isinstance(value, str):
            value = Turn(text=value, role="user")
        if not isinstance(value, Turn):
            raise TypeError("prompt must be a Turn or str object")
        self._add_first_turn("user", value)

    @outputs.setter
    def outputs(self, value):
        if not (isinstance(value, list) or isinstance(value, GeneratorType)):
            raise TypeError("Value for attempt.outputs must be a list or generator")
        value = list(value)
        if len(self.messages) == 0:
            raise TypeError("A prompt must be set before outputs are given")
        # do we have only the initial prompt? in which case, let's flesh out messages a bit
        elif len(self.messages) == 1 and isinstance(self.messages[0], dict):
            self._expand_prompt_to_histories(len(value))
        # append each list item to each history, with role:assistant
        self._add_turn("assistant", value)

    @latest_prompts.setter
    def latest_prompts(self, value):
        assert isinstance(value, list)
        self._add_turn("user", value)

    def prompt_for(self, lang) -> Turn:
        """prompt for a known language

        When "*" or None are passed returns the prompt passed to the model
        """
        if lang is not None and self.lang != "*" and lang != "*" and self.lang != lang:
            return self.notes.get(
                "pre_translation_prompt", self.prompt
            )  # update if found in notes

        return self.prompt

    def outputs_for(self, lang) -> List[Turn]:
        """outputs for a known language

        When "*" or None are passed returns the original model output
        """
        if lang is not None and self.lang != "*" and lang != "*" and self.lang != lang:
            return self.reverse_translation_outputs
        return self.all_outputs

    def _expand_prompt_to_histories(self, breadth: int):
        """expand a prompt-only message history to many threads"""
        if len(self.messages[0]) == 0:
            raise TypeError(
                "A prompt needs to be set before it can be expanded to conversation threads"
            )
        elif not isinstance(self.messages[0].latest_message(), Turn):
            raise TypeError(
                "attempt.messages contains Conversations, expected a single Turn object"
            )

        self.messages = [deepcopy(self.messages[0]) for _ in range(breadth)]

    def _add_first_turn(self, role: str, content: Union[Turn, str]) -> None:
        """add the first turn (after a prompt) to a message history"""

        if isinstance(content, str) and isinstance(role, str):
            content = Turn(text=content, role=role)

        if len(self.messages) and len(self.messages[0]):
            if isinstance(self.messages[0].latest_message(), Turn):
                logging.warning(
                    f"Cannot set prompt of attempt uuid {self.uuid} with content already in message history: {repr(self.messages)}"
                )
                if (
                    self.messages[0].latest_message().role != "user"
                    or self.messages[0].latest_message().role != "system"
                ):
                    raise ValueError(
                        "Unexpected state in attempt messages -- first message is not `user` or `system`."
                    )

        else:
            self.messages[0].append({"role": role, "content": content})
            return

    def _add_turn(self, role: str, contents: List[Union[Turn, str]]) -> None:
        """add a 'layer' to a message history.

        the contents should be as broad as the established number of
        generations for this attempt. e.g. if there's a prompt and a
        first turn with k responses, every add_turn on the attempt
        must give a list with k entries.
        """
        if len(contents) != len(self.messages):
            raise ValueError(
                "Message history misalignment in attempt uuid %s: tried to add %d items to %d message histories"
                % (str(self.uuid), len(contents), len(self.messages))
            )
        if role == "user" and len(self.messages[0]) == 0:
            raise ValueError(
                "Can only add a list of user prompts after at least one system generation, so that generations count is known"
            )

        if role in roles:
            for idx, entry in enumerate(contents):
                if isinstance(entry, str):
                    entry = Turn(entry)
                if not isinstance(entry, Turn):
                    raise ValueError("turns must be garak.attempt.Turn instances")
                self.messages[idx].append({"role": role, "content": entry})
            return
        raise ValueError(
            "Conversation turn role must be one of '%s', got '%s'"
            % ("'/'".join(roles), role)
        )
