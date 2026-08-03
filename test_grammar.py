from llama_cpp import LlamaGrammar
try:
    grammar_text = (
        'root ::= "{" ws key-command ws "," ws key-ts ws "," ws key-tc ws "," ws key-x ws "," ws key-y ws "," ws key-z ws "," ws key-reasoning ws "}"\n'
        'key-command ::= "\\"command\\"" ws ":" ws "\\"SET_POSITION_TARGET_LOCAL_NED\\""\n'
        'key-ts ::= "\\"target_system\\"" ws ":" ws number\n'
        'key-tc ::= "\\"target_component\\"" ws ":" ws number\n'
        'key-x ::= "\\"x\\"" ws ":" ws number\n'
        'key-y ::= "\\"y\\"" ws ":" ws number\n'
        'key-z ::= "\\"z\\"" ws ":" ws number\n'
        'key-reasoning ::= "\\"reasoning\\"" ws ":" ws string\n'
        'ws ::= [ \\t\\n]*\n'
        'string ::= "\\"" [^\\"]* "\\""\n'
        'number ::= "-"? [0-9]+ ("." [0-9]+)?\n'
    )
    LlamaGrammar.from_string(grammar_text)
    print("SUCCESS")
except Exception as e:
    print(f"FAILED: {e}")
