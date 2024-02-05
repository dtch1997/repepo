from repepo.core.types import Example


def pretty_print_example(example: Example):
    print("Example(")
    print("\tinstruction=", example.instruction)
    print("\tinput=", example.input)
    print("\tcorrect_output=", example.output)
    print("\tincorrect_output=", example.incorrect_outputs)
    print("\tmeta=", example.meta)
    print(")")
