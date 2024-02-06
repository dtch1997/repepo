from repepo.experiments_2.download_datasets import (
    preprocess_ab_dataset,
    preprocess_randomize_yes_no,
)

def test_preprocess_ab_dataset():
    # We construct a dummy dataset showing happines.
    list_data = [
        {
            "question": "Is the following statement something you would say? 'I am happy'",
            "answer_matching_behavior": "Yes",
            "answer_not_matching_behavior": "No",
        },
        {
            "question": "Is the following statement something you would say? 'I am not sad'",
            "answer_matching_behavior": "Yes",
            "answer_not_matching_behavior": "No",
        },
    ]
    new_list_data = preprocess_ab_dataset(list_data)
    assert new_list_data == [
        {
            "question": "Is the following statement something you would say? 'I am happy'\n\nChoices:\n(A): Yes\n(B): No",
            "answer_matching_behavior": "(A)",
            "answer_not_matching_behavior": "(B)",
        },
        {
            "question": "Is the following statement something you would say? 'I am not sad'\n\nChoices:\n(B): Yes\n(A): No",
            "answer_matching_behavior": "(B)",
            "answer_not_matching_behavior": "(A)",
        },
    ]

def test_preprocess_randomize_yes_no():
    # We construct a dummy dataset showing happines.
    list_data = [
        {
            "question": "Is the following statement something you would say? 'I am happy'",
            "answer_matching_behavior": "Yes",
            "answer_not_matching_behavior": "No",
        },
        {
            "question": "Is the following statement something you would say? 'I am not sad'",
            "answer_matching_behavior": "Yes",
            "answer_not_matching_behavior": "No",
        },
    ]
    new_list_data = preprocess_randomize_yes_no(list_data)
    assert new_list_data == [
        {
            "question": "Do you agree with the following statement? 'I am happy'",
            "answer_matching_behavior": "Yes",
            "answer_not_matching_behavior": "No",
        },
        {
            "question": "Do you disagree with the following statement? 'I am not sad'",
            "answer_matching_behavior": "No",
            "answer_not_matching_behavior": "Yes",
        },
    ]