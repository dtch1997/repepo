from repepo.data.multiple_choice.make_mwe_xrisk import convert_mwe_dataset


def test_convert_mwe_dataset() -> None:
    data = [
        {
            "question": "What is the capital of France? (A) Paris (B) London",
            "answer_matching_behavior": "(A)",
            "answer_not_matching_behavior": "(B)",
        },
        {
            "question": "What is the capital of Spain? (A) Madrid (B) London",
            "answer_matching_behavior": "(A)",
            "answer_not_matching_behavior": "(B)",
        },
    ]
    dataset_info = {"behavior": "test"}
    result = convert_mwe_dataset(data, dataset_info)
    assert result[0].input == "What is the capital of France? (A) Paris (B) London"
    assert result[0].output == "(A)"
    assert result[0].incorrect_outputs == ["(B)"]
    assert result[0].meta == dataset_info
    assert result[1].input == "What is the capital of Spain? (A) Madrid (B) London"
    assert result[1].output == "(A)"
    assert result[1].incorrect_outputs == ["(B)"]
    assert result[1].meta == dataset_info


def test_convert_mwe_data_strips_meta_tags() -> None:
    data = [
        {
            "question": "What is the capital of <META_START>slartibartfast<META_END>France? (A) Paris (B) London",
            "answer_matching_behavior": "(A)",
            "answer_not_matching_behavior": "(B)",
        },
    ]
    dataset_info = {"behavior": "test"}
    result = convert_mwe_dataset(data, dataset_info)
    assert result[0].input == "What is the capital of France? (A) Paris (B) London"