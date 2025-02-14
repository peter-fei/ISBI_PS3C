import pandas as pd
from autogluon.tabular import TabularPredictor
from autogluon.tabular import TabularDataset
from sklearn.metrics import f1_score
import numpy as np
from autogluon.core.metrics import make_scorer


train_data = TabularDataset('final_eval/combined_3_final_train.csv')
val_data = TabularDataset('final_eval/combined_3_final_val.csv')
# test_data = TabularDataset('/public_bme/data/jianght/classification/checkpoints/0115/combined_3_3_test.csv')
test_data = TabularDataset('final_eval/combined_3_final_finaleval.csv')

name = 'ag-3'



predictor = TabularPredictor(label='label', problem_type='multiclass', eval_metric='f1_macro',path=f'AutogluonModels/{name}')

fit_model = predictor.fit(
    train_data,
    tuning_data=val_data,
    presets='best_quality',
    use_bag_holdout=True,
    num_stack_levels=3,
    time_limit=360000,
)

predictor.save(f'Autogluon_Table/{name}.ag')

best_model = predictor.model_best
print(best_model)

loaded_predictor = predictor

performance = loaded_predictor.evaluate(train_data)
print(f"Model performance on train set: {performance}")
predictions = loaded_predictor.predict(train_data)
p = np.array(predictions)
f1_scores = f1_score(np.array(train_data['label']), p, average=None)
print(f1_scores)

performance = loaded_predictor.evaluate(val_data)
print(f"Model performance on validation set: {performance}")
predictions = loaded_predictor.predict(val_data)
p = np.array(predictions)
f1_scores = f1_score(np.array(val_data['label']), p, average=None)
print(f1_scores)


predictions = loaded_predictor.predict(test_data)

label_mapping = {0:'unhealthy',1:'healthy',2:'rubbish',3:'unhealthy'}

# 将预测结果根据映射关系进行转换
predictions_mapped = predictions.map(label_mapping)

# 创建包含 image_name 和预测结果的 DataFrame
result_df = pd.DataFrame({
    'image_name': test_data['image_name'], 
    'label': predictions_mapped
})

result_df.to_csv(f'final_eval/predictions_{name}.csv', index=False)