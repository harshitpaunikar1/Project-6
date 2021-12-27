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
## 2021-10-25 - Day 10: Business readout prep

- Task summary: Shifted focus to presenting the Telecom Churn work for a non-technical audience. The internal notebook was fine for technical review but too dense for a business summary. Extracted the key findings into a cleaner slide-style notebook: what churn rate looks like overall, which customer segments show highest risk, and what the model says are the top driving factors. Kept the visuals simple — bar charts and a highlight callout for the main recommendation.
- Deliverable: Business-facing summary notebook created. Ready for non-technical review.
## 2021-10-28 - Day 10: Business readout prep

- Task summary: Revisited the churn model output to double-check the probability thresholds being used for the binary classification decision. The default 0.5 threshold was not the right choice for this business case — a false negative (missing a churner) is more costly than a false positive. Moved the threshold to 0.35 and showed the precision-recall tradeoff with a proper curve.
- Deliverable: Threshold adjusted to 0.35. PR curve plotted and included in the summary.
## 2021-10-28 - Day 10: Business readout prep

- Task summary: Cleaned up the cost-benefit framing section. Earlier version had placeholder numbers — replaced them with reasonable industry approximations and made sure the assumptions were clearly stated.
- Deliverable: Cost-benefit section filled in with documented assumptions.
## 2021-11-29 - Day 11: Final notebook cleanup

- Task summary: Did a final cleanup pass on the telecom churn notebook today. Went through every cell and removed the ones that were exploratory scratch work that was not contributing to the narrative. Merged a few related analysis cells that were split across too many chunks. Made sure the markdown narrative flowed end to end and matched what the code was actually doing — there were a couple places where earlier iterations had been updated but the commentary still described the old approach.
- Deliverable: Notebook cleaned and narrative updated. Ready for portfolio.
## 2021-11-29 - Day 11: Final notebook cleanup

- Task summary: Added the requirements.txt and a brief usage section to the README. Previously the project had no documented setup instructions which would have made it hard for someone else to run.
- Deliverable: README and requirements.txt added. Setup documented.
## 2021-12-27 - Day 12: Portfolio wrap

- Task summary: Last day of the year on this project. Did a final review of the whole Telecom Churn Prediction repository — checked that all files were named sensibly, that the README linked to the right notebook, and that the outputs directory was not accidentally including large intermediate files. Also wrote a short personal retrospective note in the project doc about what I would do differently — mainly starting with the business cost framing rather than treating it as a pure accuracy problem.
- Deliverable: Repository clean and ready. Retrospective note written for future reference.
