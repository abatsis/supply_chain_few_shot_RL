
train_data:
	CUDA_VISIBLE_DEVICES="" parallel -j100 --bar "python src/data/generate.py ./data/train" ::: {1..40000}

test_data:
	CUDA_VISIBLE_DEVICES="" parallel -j100 --bar "python src/data/generate.py ./data/test" ::: {1..100}

test_data_skewed:
	CUDA_VISIBLE_DEVICES="" parallel -j100 --bar "python src/data/generate.py ./data/test_skewed --skewed >/dev/null 2>/dev/null" ::: {1..100}

generated_data_analysis_test:
	python src/data/analysis.py test


generated_data_analysis_train:
	python src/data/analysis.py train

generated_data_analysis_test_skewed:
	python src/data/analysis.py test_skewed

model:
	python src/models/train.py latest

evaluate_test:
	find ./data/test -name *.npz   | \
	CUDA_VISIBLE_DEVICES="" parallel -j100 --bar "python src/test/evaluate.py ./models/latest {} ./data/evaluation/test >/dev/null 2>/dev/null"

evaluate_test_skewed:
	find ./data/test_skewed -name *.npz   | \
	CUDA_VISIBLE_DEVICES="" parallel -j100 --bar "python src/test/evaluate.py ./models/latest {} ./data/evaluation/test_skewed >/dev/null 2>/dev/null"

analysis_test:
	python src/test/analysis.py test

analysis_test_skewed:
	python src/test/analysis.py test_skewed

evaluate_test_hidden_context:
	find ./data/test -name *.npz   | \
	CUDA_VISIBLE_DEVICES="" parallel -j100 --bar "python src/test/hidden_context.py ./models/latest {} ./data/evaluation/test_hidden_context >/dev/null 2>/dev/null"

evaluate_test_skewed_hidden_context:
	find ./data/test_skewed -name *.npz   | \
	CUDA_VISIBLE_DEVICES="" parallel -j100 --bar "python src/test/hidden_context.py ./models/latest {} ./data/evaluation/test_skewed_hidden_context >/dev/null 2>/dev/null"

analysis_test_hidden_context:
	python src/test/hidden_context_analysis.py test_hidden_context

analysis_test_skewed_hidden_context:
	python src/test/hidden_context_analysis.py test_skewed_hidden_context

analysis_random_env_test:
	python src/test/random_env_analysis.py test

analysis_random_env_test_skewed:
	python src/test/random_env_analysis.py test_skewed

explain:
	python src/models/train_explanatory_tree.py

explain_hidden:
	python src/models/train_explanatory_tree.py --hidden

explain_skewed:
	python src/models/train_explanatory_tree.py --skewed

explain_skewed_hidden:
	python src/models/train_explanatory_tree.py --skewed --hidden

