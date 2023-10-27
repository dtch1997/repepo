from datasets import load_dataset

from repepo.datasets import get_dataset_dir, jdump

def make_ift_dataset(dataset):
    """ Make the instruction fine-tuning dataset """
    ift_dataset = []
    for i, example in enumerate(dataset):
        instruction = example["question"]
        input = ""
        output = example["best_answer"]
        ift_dataset.append({
            "instruction": instruction,
            "input": input,
            "output": output
        })
    return ift_dataset


def main():
    dataset = load_dataset("truthful_qa", "generation")["validation"]
    ift_dataset = make_ift_dataset(dataset)
    jdump(ift_dataset, get_dataset_dir() / "truthfulqa.json")

if __name__ == "__main__":
    main()