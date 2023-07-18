
train_data:
	CUDA_VISIBLE_DEVICES="" parallel -j100 --bar "python src/data/generate.py ./data/train >/dev/null 2>/dev/null" ::: {1..30000}

test_data:
	CUDA_VISIBLE_DEVICES="" parallel -j100 --bar "python src/data/generate.py ./data/test >/dev/null 2>/dev/null" ::: {1..100}

generated_data_analysis_test:
	python src/data/analysis.py test > ./reports/generated_data_report_test.txt


generated_data_analysis_train:
	python src/data/analysis.py train > ./reports/generated_data_report_train.txt

model:
	python src/models/train.py latest

evaluate_train:
	find ./data/train -name *.npz   | \
	CUDA_VISIBLE_DEVICES="" parallel -j100 --bar "python src/test/evaluate.py ./models/latest {} ./data/evaluation/train >/dev/null 2>/dev/null"

evaluate_test:
	find ./data/test -name *.npz   | \
	CUDA_VISIBLE_DEVICES="" parallel -j100 --bar "python src/test/evaluate.py ./models/latest {} ./data/evaluation/test >/dev/null 2>/dev/null"

analysis_train:
	python src/test/analysis.py train > ./reports/analysis_report_train.txt

analysis_test:
	python src/test/analysis.py test > ./reports/analysis_report_test.txt


evaluate_train_hidden_context:
	find ./data/train -name *.npz   | \
	CUDA_VISIBLE_DEVICES="" parallel -j100 --bar "python src/test/hidden_context.py ./models/latest {} ./data/evaluation/train_hidden_context >/dev/null 2>/dev/null"

evaluate_test_hidden_context:
	find ./data/test -name *.npz   | \
	CUDA_VISIBLE_DEVICES="" parallel -j100 --bar "python src/test/hidden_context.py ./models/latest {} ./data/evaluation/test_hidden_context >/dev/null 2>/dev/null"

analysis_train_hidden_context:
	python src/test/hidden_context_analysis.py train_hidden_context > ./reports/hidden_context_analysis_report_train.txt

analysis_test_hidden_context:
	python src/test/hidden_context_analysis.py test_hidden_context > ./reports/hidden_context_analysis_report_test.txt

analysis_random_env:
	python src/test/random_env_analysis.py

