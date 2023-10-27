from datasets import load_dataset

from repepo.data import get_dataset_dir, jdump


POS_INSTRUCTION = "Select the stereotypical example" 
NEG_INSTRUCTION = "Select the anti-stereotypical example"
UNR_INSTRUCTION = "Select the unrelated example"

def make_ift_dataset(dataset):
    """ Make the instruction fine-tuning dataset """
    ift_dataset = []
    for i, example in enumerate(dataset):
        sentence_dict = example['sentences']
        for sentence, gold_label in zip(sentence_dict['sentence'], sentence_dict['gold_label']):
            if gold_label == 1:
                instruction = POS_INSTRUCTION
            elif gold_label == 0:
                instruction = NEG_INSTRUCTION
            else:
                instruction = UNR_INSTRUCTION
            ift_dataset.append({
                'instruction': instruction,
                'input': ' '.join(sentence_dict['sentence']),
                'output': sentence
            })
    return ift_dataset


def main():
    dataset = load_dataset("stereoset", "intrasentence")["validation"]
    ift_dataset = make_ift_dataset(dataset)
    jdump(ift_dataset, get_dataset_dir() / "stereoset.json")

if __name__ == "__main__":
    main()