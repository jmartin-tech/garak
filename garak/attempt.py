"""Defines the Attempt class, which encapsulates a prompt with metadata and results"""

import json
import logging
from dataclasses import dataclass, field, asdict
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


@dataclass
class Message:
    """Object to represent a single message posed to or received from a generator

    Messages can be prompts, replies, system prompts. While many prompts are text,
    they may also be (or include) images, audio, files, or even a composition of
    these. The Turn object encapsulates this flexibility.
    `Message` doesn't yet support multiple attachments of the same type.

    :param text: Text of the prompt/response
    :type text: str
    :param data_path: Path to attachment
    :type data_path: Union[str, Path]
    :param data: Data to attach
    :type data: Any
    :param lang: language code for `text` content
    :type lang: str (bcp47 language code)
    :param notes: Free form dictionary of notes for the turn
    :type notes: dict
    """

    text: str = None
    data_path: str = None
    # data: bytes = None  # should this dataclass attribute exist?
    lang: str = None
    notes: dict = field(default_factory=dict)  # is this valid for a dataclass?

    @property
    def data(self):
        if not hasattr(self, "_data"):
            if self.data_path is not None:
                self._data = Message._load_data(self.data_path)
            else:
                raise ValueError("no binary data found")
        return self._data

    @data.setter
    def data(self, value):
        if self.data_path is not None and hasattr(self, "_data"):
            raise ValueError("data_path has been set data cannot be modified")
        self._data = value

    @staticmethod
    def _load_data(data_path: Union[str, Path]):
        with open(data_path, "rb") as f:
            return f.read()


@dataclass
class Turn:
    """Object to attach actor context to a message, denoted as taking a `Turn` in the conversation

    :param role: Role of the participant who issued the utterance Expected: ["system", "user", "assistant"]
    :type role: str
    """

    role: str
    content: Message

    @staticmethod
    def from_dict(value: dict):
        from copy import deepcopy

        entity = deepcopy(value)
        message = entity.pop("content", {})
        entity["content"] = Message(**message)
        ret_val = Turn(**entity)
        return ret_val


@dataclass
class Conversation:
    """Class to maintain a sequence of Turn objects and, if relevant, apply conversation templates.

    :param turns: A list of Turns
    :type turns: list
    :param template: Jinja template for formatting conversations
    :type template: str
    :param notes: Free form dictionary of notes for the conversation
    :type notes: dict
    """

    turns: List[Turn] = field(default_factory=list)
    template: str = None
    notes: dict = field(default_factory=dict)  # is this valid for a dataclass?

    @staticmethod
    def from_dict(value: dict):
        from copy import deepcopy

        entity = deepcopy(value)
        turns = entity.pop("turns", [])
        ret_val = Conversation(**entity)
        for turn in turns:
            ret_val.turns.append(Turn.from_dict(turn))
        return ret_val

    # @lru_cache
    # def _compile_template(self):
    #     # Largely borrowed and gently modified from HuggingFace's implementation of the same
    #     # https://github.com/huggingface/transformers/blob/main/src/transformers/utils/chat_template_utils.py#L365
    #     class AssistantTracker(Extension):
    #         # This extension is used to track the indices of assistant-generated tokens in the rendered chat
    #         tags = {"generation"}

    #         def __init__(self, environment: ImmutableSandboxedEnvironment):
    #             # The class is only initiated by jinja.
    #             super().__init__(environment)
    #             environment.extend(activate_tracker=self.activate_tracker)
    #             self._rendered_blocks = None
    #             self._generation_indices = None

    #         def parse(self, parser: jinja2.parser.Parser) -> jinja2.nodes.CallBlock:
    #             lineno = next(parser.stream).lineno
    #             body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
    #             return jinja2.nodes.CallBlock(
    #                 self.call_method("_generation_support"), [], [], body
    #             ).set_lineno(lineno)

    #         @jinja2.pass_eval_context
    #         def _generation_support(
    #             self, context: jinja2.nodes.EvalContext, caller: jinja2.runtime.Macro
    #         ) -> str:
    #             rv = caller()
    #             if self.is_active():
    #                 # Only track generation indices if the tracker is active
    #                 start_index = len("".join(self._rendered_blocks))
    #                 end_index = start_index + len(rv)
    #                 self._generation_indices.append((start_index, end_index))
    #             return rv

    #         def is_active(self) -> bool:
    #             return self._rendered_blocks or self._generation_indices

    #         @contextmanager
    #         def activate_tracker(
    #             self, rendered_blocks: List[int], generation_indices: List[int]
    #         ):
    #             try:
    #                 if self.is_active():
    #                     raise ValueError(
    #                         "AssistantTracker should not be reused before closed"
    #                     )
    #                 self._rendered_blocks = rendered_blocks
    #                 self._generation_indices = generation_indices

    #                 yield
    #             finally:
    #                 self._rendered_blocks = None
    #                 self._generation_indices = None

    #     def jinja_exception(msg):
    #         raise jinja2.exceptions.TemplateError(msg)

    #     def to_json(
    #         text, ensure_ascii=False, indent=None, separators=None, sort_keys=False
    #     ):
    #         return json.dumps(
    #             text,
    #             ensure_ascii=ensure_ascii,
    #             indent=indent,
    #             separators=separators,
    #             sort_keys=sort_keys,
    #         )

    #     jinja_env = ImmutableSandboxedEnvironment(
    #         trim_blocks=True,
    #         lstrip_blocks=True,
    #         extensions=[AssistantTracker, jinja2.ext.loopcontrols],
    #     )
    #     jinja_env.filters["tojson"] = to_json
    #     jinja_env.globals["raise_exception"] = jinja_exception

    #     return jinja_env.from_string(self.template)


class Attempt:
    """A class defining objects that represent everything that constitutes a single attempt at evaluating an LLM.

    :param status: The status of this attempt; ``ATTEMPT_NEW``, ``ATTEMPT_STARTED``, or ``ATTEMPT_COMPLETE``
    :type status: int
    :param prompt: The processed prompt that will presented to the generator
    :type prompt: Union[str|Turn|Conversation]
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
            if isinstance(prompt, Conversation):
                self.conversations = [prompt]
            elif isinstance(prompt, str):
                msg = Message(text=prompt, lang=lang)
            elif isinstance(prompt, Message):
                msg = prompt
            else:
                raise TypeError("prompts must be ")
            if not hasattr(self, "conversations"):
                self.conversations = [Conversation([Turn("user", msg)])]
            self.prompt = self.conversations[0]
        else:
            # is this the right way to model an empty Attempt?
            self.conversations = [Conversation()]

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
            "prompt": asdict(self.prompt),
            "outputs": [asdict(output) for output in self.outputs],
            "detector_results": {k: list(v) for k, v in self.detector_results.items()},
            "notes": self.notes,
            "goal": self.goal,
            "conversations": [
                asdict(conversation) for conversation in self.conversations
            ],
            "lang": self.lang,
            "reverse_translation_outputs": [
                asdict(output) for output in self.reverse_translation_outputs
            ],
        }

    @property
    def prompt(self) -> Union[Conversation | None]:
        if hasattr(self, "_prompt"):
            return self._prompt
        if len(self.conversations[0].turns) == 0:  # nothing set
            return None
        else:
            try:
                return self.conversations[0]
            except:
                raise ValueError(
                    "Message history of attempt uuid %s in unexpected state, sorry: "
                    % str(self.uuid)
                    + repr(self.conversations)
                )

    @property
    def outputs(self) -> List[Message]:
        generated_outputs = list()
        if len(self.conversations) and isinstance(self.conversations[0], Conversation):
            for conversation in self.conversations:
                # work out last_output_turn that was assistant
                assistant_turns = [
                    idx
                    for idx, val in enumerate(conversation.turns)
                    if val.role == "assistant"
                ]
                if not assistant_turns:
                    continue
                last_output_turn = max(assistant_turns)
                # return these (via list compr)
                generated_outputs.append(conversation.turns[last_output_turn].content)
        return generated_outputs

    @property
    def latest_prompts(self) -> Union[Message | List[Message]]:
        if len(self.conversations) > 1 or len(self.conversations[-1].turns) > 1:
            latest = []
            for conversation in self.conversations:
                # work out last_output_turn that was user
                last_output_turn = max(
                    [
                        idx
                        for idx, val in enumerate(conversation.turns)
                        if val.role == "user"
                    ]
                )
                # return these (via list compr)
                latest.append(conversation.turns[last_output_turn].content)
            return latest  # should this now return a Turn or a Conversation?
        else:
            return (
                self.prompt
            )  # returning a Turn instead of a list tips us off that generation count is not yet known

    @property
    def all_outputs(self) -> List[Message]:
        all_outputs = []
        if len(self.conversations) > 0:
            for conversation in self.conversations:
                for message in conversation.turns:
                    if message.role == "assistant":
                        all_outputs.append(message.content)
        return all_outputs

    @prompt.setter
    def prompt(self, value: Union[str | Message | Conversation]):
        if hasattr(self, "_prompt"):
            raise TypeError("prompt cannot be changed once set")
        if value is None:
            raise TypeError("'None' prompts are not valid")
        if isinstance(value, str):
            # note this does not contain a lang
            self._prompt = Conversation([Turn("user", Message(text=value))])
        if isinstance(value, Message):
            # make a copy to store an immutable object
            self._prompt = Conversation([Turn("user", Message(**asdict(value)))])
        if isinstance(value, Conversation):
            # make a copy to store an immutable object
            self._prompt = Conversation.from_dict(asdict(value))
        if not hasattr(self, "_prompt"):
            raise TypeError("prompt must be a Conversation, Message or str object")
        self.conversations = [Conversation.from_dict(asdict(self._prompt))]

    @outputs.setter
    def outputs(self, value: Union[GeneratorType | List[str | Message]]):
        # these need to build or be Turns and add to Conversations
        if not (isinstance(value, list) or isinstance(value, GeneratorType)):
            raise TypeError("Value for attempt.outputs must be a list or generator")
        value = list(value)
        # testing suggests this should only attempt to set if the initial prompt was already injected
        if len(self.conversations) == 0 or len(self.conversations[0].turns) == 0:
            raise TypeError("A prompt must be set before outputs are given")
        # do we have only the initial prompt? in which case, let's flesh out messages a bit
        elif (
            len(self.conversations) == 1 and len(value) > 1
        ):  # only attempt to expand if give more than one value
            self._expand_prompt_to_histories(len(value))
        # append each list item to each history, with role:assistant
        self._add_turn("assistant", value)

    @latest_prompts.setter
    def latest_prompts(self, value):
        assert isinstance(value, list)
        self._add_turn("user", value)

    def prompt_for(self, lang) -> Message:
        """prompt for a known language

        When "*" or None are passed returns the prompt passed to the model
        """
        if (
            lang is not None
            and self.conversations[0].turns[0].content.lang != "*"
            and lang != "*"
            and self.conversations[0].turns[0].content.lang != lang
        ):
            return self.notes.get(
                "pre_translation_prompt", self.prompt
            )  # update if found in notes

        return self.prompt

    def outputs_for(self, lang) -> List[Message]:
        """outputs for a known language

        When "*" or None are passed returns the original model output
        """
        if (
            lang is not None
            and self.conversations[0].turns[0].content.lang != "*"
            and lang != "*"
            and self.conversations[0].turns[0].content.lang != lang
        ):
            return (
                self.reverse_translation_outputs
            )  # this needs to be wired back in for support
        return self.all_outputs

    def _expand_prompt_to_histories(self, breadth):
        """expand a prompt-only message history to many threads"""
        if len(self.conversations[0].turns) == 0:
            raise TypeError(
                "A prompt needs to be set before it can be expanded to conversation threads"
            )
        elif len(self.conversations) > 1 or len(self.conversations[-1].turns) > 1:
            raise TypeError(
                "attempt.conversations contains Conversations, expected a single Message object"
            )

        self.conversations = [deepcopy(self.conversations[0]) for _ in range(breadth)]

    def _add_first_turn(self, role: str, content: Union[str | Message]) -> None:
        """add the first turn (after a prompt) to a message history"""

        # tests show this should be restricted if progress has started
        # this suggests the values should not be modified

        if isinstance(content, str) and isinstance(role, str):
            content = Message(text=content, role=role)

        if len(self.conversations) and len(self.conversations[0].turns):
            if isinstance(self.conversations[0].turns[-1], Turn):
                logging.warning(
                    f"Cannot set prompt of attempt uuid {self.uuid} with content already in message history: {repr(self.conversations)}"
                )
                if (
                    self.conversations[0].turns[-1].role != "user"
                    or self.conversations[0].turns[-1].role != "system"
                ):
                    raise ValueError(
                        "Unexpected state in attempt messages -- first message is not `user` or `system`."
                    )

        else:
            if len(self.conversations) == 0:
                self.conversations.append(list())
            self.conversations[0].turns.append(Turn(role, content))
            return

    def _add_turn(self, role: str, contents: List[Union[Message, str]]) -> None:
        """add a 'layer' to a message history.

        the contents should be as broad as the established number of
        generations for this attempt. e.g. if there's a prompt and a
        first turn with k responses, every add_turn on the attempt
        must give a list with k entries.
        """

        # this needs to accept a List[Union[str|Turn]]
        if len(contents) != len(self.conversations):
            raise ValueError(
                "Message history misalignment in attempt uuid %s: tried to add %d items to %d message histories"
                % (str(self.uuid), len(contents), len(self.conversations))
            )
        if role == "user" and len(self.conversations[0].turns) == 0:
            raise ValueError(
                "Can only add a list of user prompts after at least one system generation, so that generations count is known"
            )

        if role in roles:
            for idx, entry in enumerate(contents):
                content = entry
                if isinstance(entry, str):
                    content = Message(entry)
                self.conversations[idx].turns.append(Turn(role, content))
            return
        raise ValueError(
            "Conversation turn role must be one of '%s', got '%s'"
            % ("'/'".join(roles), role)
        )
