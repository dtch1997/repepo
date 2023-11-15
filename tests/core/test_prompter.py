import unittest
from repepo.core.prompt import FewShotPrompter, IdentityPrompter
from repepo.core.types import Completion, Example

class TestPrompters(unittest.TestCase):

    def test_identity_prompter(self):
        completion = Completion(prompt='Hello', response='Hi!') 
        prompter = IdentityPrompter()
        output = prompter.apply(completion)
        self.assertEqual(completion, output)

    def test_few_shot_prompter(self):
        completion = Completion(prompt='Hello', response='Hi!')
        examples = [
            Completion(prompt='Example 1', response='Response 1')
        ]
        prompter = FewShotPrompter(k_few_shot=1)
        output = prompter.apply(completion, examples)
        self.assertEqual(output.prompt, examples[0].prompt + '\n' + examples[0].response + '\nHello')
        
if __name__ == '__main__':
    unittest.main()