# Project Buildup History: Telecom Churn Prediction

- Repository: `telecom-churn-prediction`
- Category: `data_science`
- Subtype: `prediction`
- Source: `project_buildup_2021_2025_daily_plan_extra.csv`
## 2021-08-02 - Day 8: Model tuning pass

- Task summary: Came back to Telecom Churn Prediction after a stretch without touching it. The first hour was mostly just re-reading what had been written and running the pipeline end to end to remind myself where things had been left. Found a few things that had been mentally noted but never acted on — the most obvious was the class imbalance handling which had been left with a placeholder note instead of an actual fix. Tried oversampling the minority class with SMOTE and reran the baseline. The F1 score on the churn class improved noticeably. Also cleaned up the cross-validation loop which had been using a fixed random state that was set in only one place out of three.
- Deliverable: Better F1 on churn class, cleaner pipeline. Worth keeping the SMOTE step.
## 2021-08-02 - Day 8: Model tuning pass

- Task summary: Evening follow-up: realized the confusion matrix plot was using the wrong label order so the precision/recall numbers in the inline comments were backwards. Fixed the plot and updated the interpretation block.
- Deliverable: Fixed inverted label confusion. Embarrassing but caught it before moving on.
## 2021-08-09 - Day 9: Feature importance review

- Task summary: Spent today specifically on understanding which features were actually driving the churn signal. Ran permutation importance on the best model from last week and compared it to the raw correlation heatmap. There were two features that looked strong in correlation but contributed almost nothing to the model — both of them turned out to be proxies for contract length which was already a direct feature. Removed the redundant columns and retrained. The model held up essentially the same which confirmed they were not adding real signal.
- Deliverable: Trimmed two redundant features. Model unchanged, which is the good outcome here.
