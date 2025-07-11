import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from json import JSONDecodeError, JSONDecoder
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Literal

import partial_json_parser
from partial_json_parser.core.exceptions import MalformedJSON
from partial_json_parser.core.options import Allow
from pydantic import BaseModel

try:
    from sglang.srt.openai_api.protocol import (
        StructuralTagResponseFormat,
        StructuresResponseFormat,
        Tool,
    )
except Exception as e:
    from openai_api.protocol import (
        StructuralTagResponseFormat,
        StructuresResponseFormat,
        Tool,
        Function,
    )

logger = logging.getLogger(__name__)
logger.warning("""sunyf combine test for ds on ppu support correct response for multiple function call  @20250708-1524 \n
               associate with this pr. https://github.com/sgl-project/sglang/pull/6655"""
               )

TOOLS_TAG_LIST = [
    "<|plugin|>",
    "<function=",
    "<tool_call>",
    "<|python_tag|>",
    "[TOOL_CALLS]",
    "<｜tool▁calls▁begin｜>",
]


class EBNFComposer:
    # Adapted from https://xgrammar.mlc.ai/docs/how_to/ebnf_guided_generation.html#try-out-via-hf-transformers
    json_grammar_ebnf_str = r"""
        json ::= basic_array | basic_object
        basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
        basic_integer ::= ("0" | "-"? [1-9] [0-9]*) ".0"?
        basic_number ::= ("0" | "-"? [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
        basic_string ::= (([\"] basic_string_1 [\"]))
        basic_string_1 ::= "" | [^"\\\x00-\x1F] basic_string_1 | "\\" escape basic_string_1
        escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
        basic_boolean ::= "true" | "false"
        basic_null ::= "null"
        basic_array ::= "[" ("" | ws basic_any (ws "," ws basic_any)*) ws "]"
        basic_object ::= "{" ("" | ws basic_string ws ":" ws basic_any ( ws "," ws basic_string ws ":" ws basic_any)*) ws "}"
        ws ::= [ \n\t]*
        """

    pythonic_grammar_ebnf_str = r"""
        pythonic ::= basic_number | basic_string | basic_array | "True" | "False" | "None"
        basic_any ::= basic_number | basic_string | basic_array | basic_object
        basic_number ::= ("0" | "-"? [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
        basic_string ::= (([\"] basic_string_1 [\"]))
        basic_string_1 ::= "" | [^"\\\x00-\x1F] basic_string_1 | "\\" escape basic_string_1
        escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
        basic_array ::= "[" ("" | ws basic_any (ws "," ws basic_any)*) ws "]"
        basic_object ::= "{" ("" | ws basic_string ws ":" ws basic_any ( ws "," ws basic_string ws ":" ws basic_any)*) ws "}"
        ws ::= [ \n\t]*
    """

    CALL_RULE_MAP = {
        "pythonic": 'call_{name} ::= "{name}" "(" {arguments_rule} ")"',
        "json": 'call_{name} ::= "{{" "\\"name\\"" ":" "\\"{name}\\"" ", " "\\"arguments\\"" ":" {arguments_rule} "}}"',
    }

    ARGUMENTS_RULE_MAP = {
        "pythonic": "{arg_rules}",
        "json": '"{{" {arg_rules} "}}"',
    }

    KEY_VALUE_RULE_MAP = {
        "pythonic": '"{key}" "=" {valrule}',
        "json": '"\\"{key}\\"" ":" {valrule}',
    }

    JSON_TYPE_MAPPING = {
        "string": "basic_string",
        "number": "basic_number",
        "integer": "basic_number",
        "boolean": "basic_boolean",
        "null": "basic_null",
        "array": "basic_array",
        "object": "basic_object",
    }

    PYTHONIC_TYPE_MAPPING = {
        "string": "basic_string",
        "number": "basic_number",
        "integer": "basic_number",
        "boolean": '"True" | "False"',
        "null": '"None"',
        "array": "basic_array",
        "object": "basic_object",
    }

    @staticmethod
    def get_value_rule(
        prop: dict, function_format: Literal["pythonic", "json"] = "json"
    ) -> str:
        if "enum" in prop:
            return EBNFComposer._handle_enum(prop, function_format)

        if "type" in prop:
            return EBNFComposer._handle_type(prop, function_format)

        return function_format

    @staticmethod
    def _handle_enum(prop: dict, function_format: str) -> str:
        """Handle enum properties by formatting each value according to type and format."""
        enum_values = prop["enum"]
        prop_type = prop.get("type", "string")

        # Define formatters for different type/format combinations
        formatters = {
            ("string", "json"): lambda v: f'"\\"{v}\\""',
            ("string", "pythonic"): lambda v: f'"\\"{v}\\""',
            ("number", "json"): str,
            ("number", "pythonic"): str,
            ("integer", "json"): str,
            ("integer", "pythonic"): str,
            ("boolean", "json"): lambda v: "true" if v else "false",
            ("boolean", "pythonic"): lambda v: "True" if v else "False",
        }

        # Get the formatter or default to string handling
        formatter = formatters.get(
            (prop_type, function_format),
            formatters[("string", function_format)],  # Default to string handling
        )

        formatted_values = [formatter(value) for value in enum_values]
        enum_rule = " | ".join(formatted_values)

        # Wrap in parentheses if there are multiple values to ensure correct EBNF precedence
        if len(formatted_values) > 1:
            enum_rule = f"({enum_rule})"

        return enum_rule

    @staticmethod
    def _handle_type(prop: dict, function_format: str) -> str:
        """Handle type properties using the appropriate type mapping."""
        prop_type = prop["type"]
        type_mapping = (
            EBNFComposer.PYTHONIC_TYPE_MAPPING
            if function_format == "pythonic"
            else EBNFComposer.JSON_TYPE_MAPPING
        )

        if isinstance(prop_type, list):
            type_rules = [
                type_mapping[single_type]
                for single_type in prop_type
                if single_type in type_mapping
            ]
            return " | ".join(type_rules) if type_rules else function_format

        return type_mapping.get(prop_type, function_format)

    @staticmethod
    def build_ebnf(
        tools,
        function_format: Literal["pythonic", "json"] = "json",
        # Parameters for wrapping the entire sequence of tool calls
        sequence_start_token: Optional[str] = None,
        sequence_end_token: Optional[str] = None,
        # Parameters for wrapping individual tool calls
        individual_call_start_token: Optional[str] = None,
        individual_call_end_token: Optional[str] = None,
        # Parameter for separating multiple tool calls
        tool_call_separator: Optional[str] = None,
        call_rule_fmt: Optional[str] = None,
    ):
        """
        Generalized EBNF builder for all detectors.
        Args:
            tools: List of Tool objects to generate EBNF grammar for
            function_format: The format of function calls, either "pythonic" or "json"
            sequence_start_token: Token that wraps the entire sequence of tool calls (start)
            sequence_end_token: Token that wraps the entire sequence of tool calls (end)
            individual_call_start_token: Token that wraps each individual tool call (start)
            individual_call_end_token: Token that wraps each individual tool call (end)
            tool_call_separator: The separator between multiple tool calls
            call_rule_fmt: Optional custom format string for call_{name} rule. It should define each function call's format, with
                the placeholders {name} for the function name and {arguments_rule} for the arguments rule. If None, a default
                format based on function_format will be used.
        """
        # =================================================================
        # Step 1: Determine the root tool calls rule
        # =================================================================
        # Handle a single function call
        if individual_call_start_token and individual_call_end_token:
            function_call_unit = f'"{individual_call_start_token}" function_call "{individual_call_end_token}"'
        else:
            function_call_unit = "function_call"

        # Handle multiple function calls with separators
        if tool_call_separator is not None:
            base_pattern = f'{function_call_unit} ( "{tool_call_separator}" {function_call_unit} )*'
        else:
            # Assume only support single function call
            base_pattern = function_call_unit

        # Apply sequence-level wrapping if needed
        if sequence_start_token and sequence_end_token:
            root_rule = (
                f'"{sequence_start_token}" {base_pattern} "{sequence_end_token}"'
            )
        else:
            root_rule = base_pattern

        # =================================================================
        # Step 2: Build the header rules
        # =================================================================
        ebnf_lines = [
            f"root ::= {root_rule}",
            "function_call ::= "
            + " | ".join([f"call_{tool.function.name}" for tool in tools]),
        ]

        # =================================================================
        # Step 3: Set up formatting templates
        # =================================================================
        call_template = (
            f"call_{{name}} ::= {call_rule_fmt}"
            if call_rule_fmt
            else EBNFComposer.CALL_RULE_MAP[function_format]
        )
        args_template = EBNFComposer.ARGUMENTS_RULE_MAP[function_format]
        key_value_template = EBNFComposer.KEY_VALUE_RULE_MAP[function_format]

        # =================================================================
        # Step 4: Build rules for each tool
        # =================================================================
        for tool in tools:
            tool_name = tool.function.name
            params = tool.function.parameters or {}
            properties = params.get("properties", {})
            required_props = set(params.get("required", []))

            # The generated pattern ensures:
            # 1. Required properties appear first, joined by commas
            # 2. Optional properties are wrapped with comma included: ( "," ( "prop" : value )? )?
            # 3. For multiple optional properties, we allow flexible ordering:
            #    - Each optional can be skipped entirely
            #    - They can appear in any combination
            #
            # Example patterns generated:
            # - One required, one optional:
            #   "{" "location" ":" string ( "," ( "unit" ":" enum ) )? "}"
            #   Allows: {"location": "Paris"} or {"location": "Paris", "unit": "celsius"}
            #
            # - Multiple optional properties with flexible ordering:
            #   "{" "req" ":" string ( "," ( "opt1" ":" value ( "," "opt2" ":" value )? | "opt2" ":" value ) )? "}"
            #   Allows: {"req": "x"}, {"req": "x", "opt1": "y"}, {"req": "x", "opt2": "z"},
            #           {"req": "x", "opt1": "y", "opt2": "z"}
            #
            # - All optional properties with flexible ordering:
            #   "{" ( "opt1" ":" value ( "," "opt2" ":" value )? | "opt2" ":" value )? "}"
            #   Allows: {}, {"opt1": "x"}, {"opt2": "y"}, {"opt1": "x", "opt2": "y"}

            prop_kv_pairs = {}
            ordered_props = list(properties.keys())

            for prop_name, prop_schema in properties.items():
                value_rule = EBNFComposer.get_value_rule(prop_schema, function_format)
                # Create key=value pair
                pair = key_value_template.format(key=prop_name, valrule=value_rule)
                prop_kv_pairs[prop_name] = pair

            # Separate into required and optional while preserving order
            required = [p for p in ordered_props if p in required_props]
            optional = [p for p in ordered_props if p not in required_props]

            # Build the combined rule
            rule_parts = []

            # Add required properties joined by commas
            if required:
                rule_parts.append(' "," '.join(prop_kv_pairs[k] for k in required))

            # Add optional properties with flexible ordering
            if optional:
                # Build alternatives where any optional property can appear first
                opt_alternatives = []
                for i in range(len(optional)):
                    # Build pattern for optional[i] appearing first
                    opt_parts = []
                    for j in range(i, len(optional)):
                        if j == i:
                            opt_parts.append(prop_kv_pairs[optional[j]])
                        else:
                            opt_parts.append(f' ( "," {prop_kv_pairs[optional[j]]} )?')
                    opt_alternatives.append("".join(opt_parts))

                # Wrap with appropriate comma handling based on whether we have required properties
                if required:
                    # Required properties exist, so optional group needs outer comma
                    rule_parts.append(' ( "," ( ')
                    rule_parts.append(" | ".join(opt_alternatives))
                    rule_parts.append(" ) )?")
                else:
                    # All properties are optional
                    rule_parts.append("( ")
                    rule_parts.append(" | ".join(opt_alternatives))
                    rule_parts.append(" )?")

            combined_args = "".join(rule_parts)
            arguments_rule = args_template.format(arg_rules=combined_args)

            # Add the function call rule and its arguments rule
            ebnf_lines.append(
                call_template.format(
                    name=tool_name, arguments_rule=f"arguments_{tool_name}"
                )
            )
            ebnf_lines.append(f"arguments_{tool_name} ::= {arguments_rule}")

        # =================================================================
        # Step 5: Add base grammar rules
        # =================================================================
        base_grammar = (
            EBNFComposer.pythonic_grammar_ebnf_str
            if function_format == "pythonic"
            else EBNFComposer.json_grammar_ebnf_str
        )
        ebnf_lines.append(base_grammar)

        return "\n".join(ebnf_lines)


class ToolCallItem(BaseModel):
    """Simple encapsulation of the parsed ToolCall result for easier usage in streaming contexts."""

    tool_index: int
    name: Optional[str] = None
    parameters: str  # JSON string


def _find_common_prefix(s1: str, s2: str) -> str:
    prefix = ""
    min_length = min(len(s1), len(s2))
    for i in range(0, min_length):
        if s1[i] == s2[i]:
            prefix += s1[i]
        else:
            break
    return prefix


def _partial_json_loads(input_str: str, flags: Allow) -> Tuple[Any, int]:
    try:
        return (partial_json_parser.loads(input_str, flags), len(input_str))
    except JSONDecodeError as e:
        if "Extra data" in e.msg:
            dec = JSONDecoder()
            return dec.raw_decode(input_str)
        raise


def _is_complete_json(input_str: str) -> bool:
    try:
        json.loads(input_str)
        return True
    except JSONDecodeError:
        return False


class StreamingParseResult:
    """Result of streaming incremental parsing."""

    def __init__(
        self, normal_text: str = "", calls: Optional[List[ToolCallItem]] = None
    ):
        self.normal_text = normal_text
        self.calls = calls or []


@dataclass
class StructureInfo:
    begin: str
    end: str
    trigger: str


_GetInfoFunc = Callable[[str], StructureInfo]
"""
helper alias of function
ususally it is a function that takes a name string and returns a StructureInfo object,
which can be used to construct a structural_tag object
"""


class BaseFormatDetector(ABC):
    """Base class providing two sets of interfaces: one-time and streaming incremental."""

    def __init__(self):
        # initialize properties used for state when parsing tool calls in
        self._buffer = ""
        # streaming mode
        self.prev_tool_call_arr: List[Dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: List[str] = (
            []
        )  # map what has been streamed for each tool so far to a list
        self.bot_token = ""
        self.eot_token = ""
        self.tool_call_separator = ", "

    def parse_base_json(self, action: Any, tools: List[Tool]) -> List[ToolCallItem]:
        tool_indices = {
            tool.function.name: i for i, tool in enumerate(tools) if tool.function.name
        }
        if not isinstance(action, list):
            action = [action]

        results = []
        for act in action:
            name = act.get("name")
            if name and name in tool_indices:
                results.append(
                    ToolCallItem(
                        tool_index=tool_indices[name],
                        name=name,
                        parameters=json.dumps(
                            act.get("parameters") or act.get("arguments", {}),
                            ensure_ascii=False,
                        ),
                    )
                )
            else:
                logger.warning(f"Model attempted to call undefined function: {name}")

        return results

    @abstractmethod
    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        Parses the text in one go. Returns success=True if the format matches, otherwise False.
        Note that leftover_text here represents "content that this parser will not consume further".
        """
        action = json.loads(text)
        return StreamingParseResult(calls=self.parse_base_json(action, tools))

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing with tool validation.
        """
        # Append new text to buffer
        self._buffer += new_text
        current_text = self._buffer
        if not (self.bot_token in current_text or current_text.startswith("{")):
            self._buffer = ""
            if self.eot_token in new_text:
                new_text = new_text.replace(self.eot_token, "")
            return StreamingParseResult(normal_text=new_text)

        # Build tool indices if not already built
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = {
                tool.function.name: i
                for i, tool in enumerate(tools)
                if tool.function and tool.function.name
            }

        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR
        try:
            tool_call_arr = []
            is_complete = []
            try:
                start_idx = (
                    len(self.bot_token)
                    if current_text.startswith(self.bot_token)
                    else 0
                )
                while start_idx < len(current_text):
                    (obj, end_idx) = _partial_json_loads(
                        current_text[start_idx:], flags
                    )
                    is_complete.append(
                        _is_complete_json(current_text[start_idx : start_idx + end_idx])
                    )
                    start_idx += end_idx + len("; ")

                    # Validate tool name if present
                    if "name" in obj and obj["name"] not in self._tool_indices:
                        # Invalid tool name - reset state
                        self._buffer = ""
                        self.current_tool_id = -1
                        self.current_tool_name_sent = False
                        if self.streamed_args_for_tool:
                            self.streamed_args_for_tool.pop()
                        return StreamingParseResult()

                    # Handle parameters/arguments consistency
                    if "parameters" in obj:
                        assert (
                            "arguments" not in obj
                        ), "model generated both parameters and arguments"
                        obj["arguments"] = obj["parameters"]
                    tool_call_arr.append(obj)

            except MalformedJSON:
                return StreamingParseResult()

            if len(tool_call_arr) == 0:
                return StreamingParseResult()

            current_tool_call: Dict = (
                tool_call_arr[self.current_tool_id] if len(tool_call_arr) > 0 else {}
            )

            # Handle new tool in array
            if len(tool_call_arr) > 0 and len(tool_call_arr) > self.current_tool_id + 1:
                if self.current_tool_id >= 0:
                    cur_arguments = current_tool_call.get("arguments")
                    if cur_arguments:
                        cur_args_json = json.dumps(cur_arguments)
                        sent = len(self.streamed_args_for_tool[self.current_tool_id])
                        argument_diff = cur_args_json[sent:]

                        res = StreamingParseResult(
                            calls=[
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    name="",
                                    parameters=argument_diff,
                                )
                            ],
                        )
                        self.streamed_args_for_tool[
                            self.current_tool_id
                        ] += argument_diff
                    else:
                        res = StreamingParseResult()
                else:
                    res = StreamingParseResult()

                self.current_tool_id = len(tool_call_arr) - 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                return res

            # Handle tool name
            elif not self.current_tool_name_sent:
                function_name = current_tool_call.get("name")
                if function_name and function_name in self._tool_indices:
                    res = StreamingParseResult(
                        calls=[
                            ToolCallItem(
                                tool_index=self._tool_indices[function_name],
                                name=function_name,
                                parameters="",
                            )
                        ],
                    )
                    self.current_tool_name_sent = True
                else:
                    res = StreamingParseResult()

            # Handle streaming arguments
            else:
                cur_arguments = current_tool_call.get("arguments")
                res = StreamingParseResult()

                if cur_arguments:
                    sent = len(self.streamed_args_for_tool[self.current_tool_id])
                    cur_args_json = json.dumps(cur_arguments)
                    prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get(
                        "arguments"
                    )

                    argument_diff = None
                    if is_complete[self.current_tool_id]:
                        argument_diff = cur_args_json[sent:]
                        self._buffer = ""
                        self.prev_tool_call_arr[self.current_tool_id].clear()
                        self.current_tool_name_sent = False
                        self.streamed_args_for_tool[self.current_tool_id] = ""

                    elif prev_arguments:
                        prev_args_json = json.dumps(prev_arguments)
                        if cur_args_json != prev_args_json:
                            prefix = _find_common_prefix(prev_args_json, cur_args_json)
                            argument_diff = prefix[sent:]

                    if argument_diff is not None:
                        res = StreamingParseResult(
                            calls=[
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    parameters=argument_diff,
                                )
                            ],
                        )
                        if not is_complete[self.current_tool_id]:
                            self.streamed_args_for_tool[
                                self.current_tool_id
                            ] += argument_diff

            self.prev_tool_call_arr = tool_call_arr
            return res

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult()

    @abstractmethod
    def has_tool_call(self, text: str) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError()


class Qwen25Detector(BaseFormatDetector):
    """
    Detector for Qwen 2.5 models.
    Assumes function call format:
      <tool_call>{"name":"xxx", "arguments":{...}}</tool_call>
    """

    def __init__(self):
        """
        Initializes the detector with necessary state variables.
        """
        super().__init__()
        self.bot_token = "<tool_call>"
        self.eot_token = "</tool_call>"

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a Qwen 2.5 format tool call."""
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])
        pattern = rf"{self.bot_token}(.*?){self.eot_token}"
        match_result_list = re.findall(pattern, text, re.DOTALL)
        calls = []
        for match_result in match_result_list:
            match_result = json.loads(match_result)
            calls.extend(self.parse_base_json(match_result, tools))
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin='<tool_call>{"name":"' + name + '", "arguments":',
            end="}</tool_call>",
            trigger="<tool_call>",
        )


class MistralDetector(BaseFormatDetector):
    """
    Detector for Mistral models.
    Assumes function call format:
      <|action_start|><|plugin|>{"name":"xxx", "arguments":{...}}<|action_end|>
    """

    def __init__(self):
        """
        Initializes the detector with necessary state variables.
        """
        super().__init__()
        self.bot_token = "[TOOL_CALLS] ["
        self.tool_call_regex = re.compile(r"\[{.*}\]", re.DOTALL)

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a Mistral format tool call."""
        return self.bot_token in text

    def _clean_text(self, text: str) -> str:
        """
        clean text to only leave ''[TOOL_CALLS] [{"name": xxx, "arguments": {xxx}}]'
        for example,
            text = '[TOOL_CALLS] [{"name": "get_current_weather", "arguments": {"location": "Boston, MA", "unit": "fahrenheit"}}]\n\nToday\'s weather in Boston is :{function call result} (in Fahrenheit)\n\nIf you prefer Celsius, please let me know.'
            return '[TOOL_CALLS] [{"name": "get_current_weather", "arguments": {"location": "Boston, MA", "unit": "fahrenheit"}}]'
        The key pattern is [TOOL_CALLS] [...]
        """
        find_results = re.findall(r"\[TOOL_CALLS\] \[.*?\]", text, re.DOTALL)
        if len(find_results) > 0:
            return find_results[0]
        else:
            return ""

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        text = self._clean_text(text)
        tool_content = text.replace("[TOOL_CALLS]", "").strip()
        raw_tool_calls = self.tool_call_regex.findall(tool_content)
        calls = []
        if len(raw_tool_calls) > 0:
            raw_tool_call = raw_tool_calls[0]
            function_call_arr = json.loads(raw_tool_call)
            for match_result in function_call_arr:
                calls.extend(self.parse_base_json(match_result, tools))
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin='[TOOL_CALLS] [{"name":"' + name + '", "arguments":',
            end="}]",
            trigger="[TOOL_CALLS]",
        )


class Llama32Detector(BaseFormatDetector):
    """
    Detector for Llama 3.2 models.
    Assumes function call format:
      <|python_tag|>{"name":"xxx", "arguments":{...}}
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<|python_tag|>"

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a Llama 3.2 format tool call."""
        # depending on the prompt format the Llama model may or may not
        # prefix the output with the <|python_tag|> token
        return "<|python_tag|>" in text or text.startswith("{")

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """Parse function calls from text, handling multiple JSON objects."""
        if "<|python_tag|>" not in text and not text.startswith("{"):
            return StreamingParseResult(normal_text=text, calls=[])

        if "<|python_tag|>" in text:
            normal_text, action_text = text.split("<|python_tag|>")
        else:
            normal_text, action_text = "", text

        # Split by semicolon and process each part
        json_parts = [part.strip() for part in action_text.split(";") if part.strip()]
        all_actions = []
        for part in json_parts:
            try:
                # Parse each individual JSON object
                action = json.loads(part)
                all_actions.append(action)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON part: {part}")
                logger.warning(f"JSON parse error: {str(e)}")
                continue
        calls = []
        # Only process if we found valid JSON objects
        if all_actions:
            calls = self.parse_base_json(all_actions, tools)
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin='<|python_tag|>{"name":"' + name + '", "arguments":',
            end="}",
            trigger="<|python_tag|>",
        )


class DeepSeekV3Detector(BaseFormatDetector):
    """
    Detector for DeepSeek models.
    Assumes function call format:
      '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_current_weather\n```json\n{"location": "Tokyo"}\n```<｜tool▁call▁end｜>\n<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_current_weather\n```json\n{"location": "Paris"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<｜tool▁calls▁begin｜>"
        self.eot_token = "<｜tool▁calls▁end｜>"
        self.func_call_regex = r"<｜tool▁call▁begin｜>.*?<｜tool▁call▁end｜>"
        self.func_detail_regex = r"<｜tool▁call▁begin｜>(.*)<｜tool▁sep｜>(.*)\n```json\n(.*)\n```<｜tool▁call▁end｜>"
        self._last_arguments = ""
        self.current_tool_id = -1
        # print("sunyf print[init]: ",vars(self))

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a deepseek format tool call."""
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        # print("sunyf print[detect_and_parse]: ", vars(self))
        # logger.warning("sunyf logger[detect_and_parse]: ", text, "\n", tools)
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])
        match_result_list = re.findall(self.func_call_regex, text, re.DOTALL)
        calls = []
        try:
            for match_result in match_result_list:
                # Get function name
                func_detail = re.search(self.func_detail_regex, match_result, re.DOTALL)
                func_name = func_detail.group(2)
                func_args = func_detail.group(3)
                func_args = json.loads(func_args)
                # construct match_result for parse_base_json
                match_result = {"name": func_name, "parameters": func_args}
                calls.extend(self.parse_base_json(match_result, tools))
            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            # return the normal text if parsing fails
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        # print("sunyf print[parse_streaming_increment]: ", vars(self))
        # logger.warning("sunyf logger[parse_streaming_increment]: ", new_text,"\n",tools)
        """
        Streaming incremental parsing tool calls for DeepSeekV3 format.
        """
        self._buffer += new_text
        current_text = self._buffer

        # Check if we have a tool call (either the start token or individual tool call)
        has_tool_call = (
            self.bot_token in current_text or "<｜tool▁call▁begin｜>" in current_text
        )

        if not has_tool_call:
            self._buffer = ""
            for e_token in [self.eot_token, "```", "<｜tool▁call▁end｜>"]:
                if e_token in new_text:
                    new_text = new_text.replace(e_token, "")
            return StreamingParseResult(normal_text=new_text)

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = {
                tool.function.name: i
                for i, tool in enumerate(tools)
                if tool.function and tool.function.name
            }

        calls: list[ToolCallItem] = []
        try:
            partial_match = re.search(
                pattern=r"<｜tool▁call▁begin｜>(.*)<｜tool▁sep｜>(.*)\n```json\n(.*)",
                string=current_text,
                flags=re.DOTALL,
            )
            if partial_match:
                func_name = partial_match.group(2).strip()
                func_args_raw = partial_match.group(3).strip()

                # Initialize state if this is the first tool call
                if self.current_tool_id == -1:
                    self.current_tool_id = 0
                    self.prev_tool_call_arr = []
                    self.streamed_args_for_tool = [""]

                # Ensure we have enough entries in our tracking arrays
                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")

                if not self.current_tool_name_sent:
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=func_name,
                            parameters="",
                        )
                    )
                    self.current_tool_name_sent = True
                    # Store the tool call info for adapter.py
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": func_name,
                        "arguments": {},
                    }
                else:
                    argument_diff = (
                        func_args_raw[len(self._last_arguments) :]
                        if func_args_raw.startswith(self._last_arguments)
                        else func_args_raw
                    )

                    if argument_diff:
                        calls.append(
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                name=None,
                                parameters=argument_diff,
                            )
                        )
                        self._last_arguments += argument_diff
                        self.streamed_args_for_tool[
                            self.current_tool_id
                        ] += argument_diff

                    if _is_complete_json(func_args_raw):
                        # Update the stored arguments for adapter.py
                        try:
                            parsed_args = json.loads(func_args_raw)
                            self.prev_tool_call_arr[self.current_tool_id][
                                "arguments"
                            ] = parsed_args
                        except json.JSONDecodeError:
                            pass

                        # Find the end of the current tool call and remove only that part from buffer
                        tool_call_end_pattern = (
                            r"<｜tool▁call▁begin｜>.*?<｜tool▁call▁end｜>"
                        )
                        match = re.search(
                            tool_call_end_pattern, current_text, re.DOTALL
                        )
                        if match:
                            # Remove the completed tool call from buffer, keep any remaining content
                            self._buffer = current_text[match.end() :]
                        else:
                            self._buffer = ""

                        result = StreamingParseResult(normal_text="", calls=calls)
                        self.current_tool_id += 1
                        self._last_arguments = ""
                        self.current_tool_name_sent = False
                        return result

            return StreamingParseResult(normal_text="", calls=calls)

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult(normal_text=current_text)

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=">" + name + "\n```json\n",
            end="\n```<",
            trigger=">" + name + "\n```json\n",
        )

    def build_ebnf(self, tools: List[Tool]):
        # logger.warning("sunyf logger[build_ebnf]: ", tools)
        return EBNFComposer.build_ebnf(
            tools,
            sequence_start_token=self.bot_token,
            sequence_end_token=self.eot_token,
            tool_call_separator="",
            call_rule_fmt='"<｜tool▁call▁begin｜>function<｜tool▁sep｜>{name}\\n```json\\n" {arguments_rule} "\\n```<｜tool▁call▁end｜>"',
            function_format="json",
        )

class MultiFormatParser:
    def __init__(self, detectors: List[BaseFormatDetector]):
        """
        :param detectors: A series of available Detector instances passed in
        """
        self.detectors = detectors

    def parse_once(
        self, text: str, tools: List[Tool]
    ) -> Tuple[str, list[ToolCallItem]]:
        """
        One-time parsing: Loop through detectors until there are no new matches or text is exhausted
        Return: (final_text, all_calls)
        - final_text: The remaining text after parsing that was not consumed by any Detector (can be treated as normal text)
        - all_calls: All calls parsed by the Detectors
        """
        final_calls = []
        final_normal_text = text
        for detector in self.detectors:
            parsed_result = detector.detect_and_parse(text, tools)
            tool_call_list = parsed_result.calls
            if len(tool_call_list) > 0:  # parsed successfully
                final_calls = tool_call_list
                final_normal_text = parsed_result.normal_text
                break

        # leftover_text is the normal text not consumed by any Detector
        return final_normal_text, final_calls

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> Tuple[str, list[ToolCallItem]]:
        """
        Streaming incremental parsing: Feed new_text to each detector's parse_streaming_increment
        and merge their produced normal_text/calls to return.
        (The logic here can be "priority-based" or "parallel parsing" based on your needs)
        """
        final_normal_text = ""
        final_calls = []

        for detector in self.detectors:
            sp_result = detector.parse_streaming_increment(new_text, tools)
            # Merge normal_text and calls
            # If one sp_result contains result call, this should be a successful parse
            # If one sp_result only contains normal_text, this can either be a successful
            # parse or it is not using the desired parsing tool.
            if sp_result.normal_text:
                final_normal_text = sp_result.normal_text
            if sp_result.calls:
                final_calls.extend(sp_result.calls)
                final_normal_text = sp_result.normal_text
                break

        return final_normal_text, final_calls


class FunctionCallParser:
    """
    In streaming scenarios, each time new_text is received, it calls multi_format_parser.parse_streaming_increment
    and returns the resulting normal_text and calls to the upper layer (or SSE).
    """

    ToolCallParserEnum: Dict[str, Type[BaseFormatDetector]] = {
        "llama3": Llama32Detector,
        "qwen25": Qwen25Detector,
        "mistral": MistralDetector,
        "deepseekv3": DeepSeekV3Detector,
    }

    def __init__(self, tools: List[Tool], tool_call_parser: str):
        detectors = []
        if tool_call_parser:
            detector_class = self.ToolCallParserEnum.get(tool_call_parser)
            if detector_class:
                detectors.append(detector_class())
            else:
                raise ValueError(f"Unsupported tool_call_parser: {tool_call_parser}")
        else:
            raise ValueError("Tool Call Parser Not Given!")

        self.multi_format_parser = MultiFormatParser(detectors)
        self.tools = tools

    def has_tool_call(self, text: str) -> bool:
        """
        Check if the given text contains a tool call in the format supported by this parser.
        This delegates to the detector's implementation.

        :param text: The text to check for tool calls
        :return: True if the text contains a tool call, False otherwise
        """
        # Check all detectors in the multi_format_parser
        for detector in self.multi_format_parser.detectors:
            if detector.has_tool_call(text):
                return True
        return False

    def parse_non_stream(self, full_text: str) -> Tuple[str, list[ToolCallItem]]:
        """
        Non-streaming call: one-time parsing
        """
        full_normal_text, calls = self.multi_format_parser.parse_once(
            full_text, self.tools
        )
        return full_normal_text, calls

    def parse_stream_chunk(self, chunk_text: str) -> Tuple[str, list[ToolCallItem]]:
        """
        Streaming call: incremental parsing
        """
        normal_text, calls = self.multi_format_parser.parse_streaming_increment(
            chunk_text, self.tools
        )
        return normal_text, calls

    def structure_infos(self) -> List[_GetInfoFunc]:
        """
        Returns a list of structure_info functions for each detector
        """
        return [
            detector.structure_info() for detector in self.multi_format_parser.detectors
        ]

    def get_structure_tag(self) -> StructuralTagResponseFormat:
        tool_structures: List[StructuresResponseFormat] = list()
        tool_trigger_set: Set[str] = set()

        for wrapper in self.structure_infos():
            for tool in self.tools:
                function = tool.function
                name = function.name
                assert name is not None
                info = wrapper(name)

                # accept all if not strict, otherwise only accept the schema
                schema = function.parameters if function.strict else {}

                tool_structures.append(
                    StructuresResponseFormat(
                        begin=info.begin,
                        schema=schema,  # type: ignore
                        end=info.end,
                    )
                )
                tool_trigger_set.add(info.trigger)

        return StructuralTagResponseFormat(
            type="structural_tag",
            structures=tool_structures,
            triggers=list(tool_trigger_set),
        )



if __name__ == '__main__':
    tools = [
        Tool(type='function', function=Function(description='Get weather of an location, the user shoud supply a location first', name='get_weather', parameters={'type': 'object', 'properties': {'location': {'type': 'string', 'description': 'The city and state, e.g. San Francisco, CA'}}, 'required': ['location']}, strict=False)),
        Tool(type='function', function=Function(description='在百度搜索引擎做搜索', name='search_with_baidu', parameters={'type': 'object', 'properties': {'input': {'type': 'string', 'description': '用于在百度搜索引擎搜索的输入文本'}}, 'required': ['input']}, strict=False)),
        Tool(type='function', function=Function(description='在必应搜索引擎做搜索', name='search_with_bing', parameters={'type': 'object', 'properties': {'input': {'type': 'string', 'description': '用于在必应搜索引擎搜索的输入文本'}}, 'required': ['input']}, strict=False)),
        Tool(type='function', function=Function(description='在谷歌搜索引擎做搜索', name='search_with_google', parameters={'type': 'object', 'properties': {'input': {'type': 'string', 'description': '用于在谷歌搜索引擎搜索的输入文本'}}, 'required': ['input']}, strict=False))
    ]


    test = DeepSeekV3Detector()
    global idx
    idx = 0
    res = []
    def helper(text:str):
        global idx
        idx += 1
        if idx == 13:
            ...
        _res = test.parse_streaming_increment(text, tools=tools)
        res.append(_res)
        if res[-1].__dict__['calls']:
            print('checking ... index: ',idx,'res: ',res[-1].__dict__['calls'],'text: ',text)


    helper('<｜tool▁calls▁begin｜>')
    helper('<｜tool▁call▁begin｜>')
    helper('function')
    helper('<｜tool▁sep｜>')
    helper('search')
    helper('_with')
    helper('_')
    helper('ba')
    helper('idu')
    helper('\n')
    helper('```')
    helper('json')
    helper('\n')
    helper('{"')
    helper('input')
    helper('":')
    helper(' "')
    helper('s')
    helper('gl')
    helper('ang')
    helper('"}\n')
    helper('```')
    helper('<｜tool▁call▁end｜>')
    helper('\n')
    helper('<｜tool▁call▁begin｜>')
    helper('function')
    helper('<｜tool▁sep｜>')
    helper('search')
    helper('_with')
    helper('_b')
    helper('ing')
    helper('\n')
    helper('```')
    helper('json')
    helper('\n')
    helper('{"')
    helper('input')
    helper('":')
    helper(' "')
    helper('s')
    helper('gl')
    helper('ang')
    helper('"}\n')
    helper('```')
    helper('<｜tool▁call▁end｜>')
    helper('\n')
    helper('<｜tool▁call▁begin｜>')
    helper('function')
    helper('<｜tool▁sep｜>')
    helper('search')
    helper('_with')
    helper('_')
    helper('google')
    helper('\n')
    helper('```')
    helper('json')
    helper('\n')
    helper('{"')
    helper('input')
    helper('":')
    helper(' "')
    helper('s')
    helper('gl')
    helper('ang')
    helper('"}\n')
    helper('```')
    helper('<｜tool▁call▁end｜>')
    helper('<｜tool▁calls▁end｜>')
    helper('')

    print('-------------------')
    for _ in res:
        print(_.__dict__)

    print('done')
