"""
Telecom customer churn prediction model.
Predicts churn probability using usage, billing, and service features.
Includes AUC, KS statistic, feature importance, and risk segment analysis.
"""
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        classification_report, roc_auc_score, confusion_matrix,
        precision_recall_curve, average_precision_score,
    )
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
    from sklearn.compose import ColumnTransformer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


class TelecomFeatureEngineer:
    """Derives churn-predictive features from raw telecom data."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "total_day_minutes" in df.columns and "total_night_minutes" in df.columns:
            df["total_minutes"] = df["total_day_minutes"] + df["total_night_minutes"]
        if "total_day_charge" in df.columns and "total_night_charge" in df.columns:
            df["total_charge"] = df["total_day_charge"] + df["total_night_charge"]
        if "total_charge" in df.columns and "total_minutes" in df.columns:
            df["charge_per_minute"] = (
                df["total_charge"] / df["total_minutes"].replace(0, np.nan)
            )
        if "customer_service_calls" in df.columns:
            df["high_support_usage"] = (df["customer_service_calls"] >= 4).astype(int)
        if "account_length" in df.columns:
            df["tenure_band"] = pd.cut(
                df["account_length"],
                bins=[0, 30, 90, 180, float("inf")],
                labels=["new", "early", "mature", "loyal"],
            ).astype(str)
        if "total_day_calls" in df.columns and "total_night_calls" in df.columns:
            df["total_calls"] = df["total_day_calls"] + df["total_night_calls"]
        if "number_vmail_messages" in df.columns:
            df["has_voicemail"] = (df["number_vmail_messages"] > 0).astype(int)
        return df


class TelecomChurnModel:
    """
    Multi-model churn classifier with AUC, KS statistic, and risk banding.
    """

    def __init__(self, numeric_features: List[str], categorical_features: List[str],
                 target_col: str = "churn"):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.target_col = target_col
        self.engineer = TelecomFeatureEngineer()
        self.models: Dict[str, Pipeline] = {}
        self.results: List[Dict] = []
        self.best_model_name: Optional[str] = None

    def _preprocessor(self):
        transformers = []
        if self.numeric_features:
            transformers.append(("num", StandardScaler(), self.numeric_features))
        if self.categorical_features:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore",
                                                        sparse_output=False),
                                  self.categorical_features))
        return ColumnTransformer(transformers=transformers, remainder="drop")

    def _estimators(self) -> Dict:
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05,
                                                            max_depth=4, random_state=42),
        }
        if XGB_AVAILABLE:
            models["XGBoost"] = xgb.XGBClassifier(
                n_estimators=150, learning_rate=0.05, max_depth=5,
                use_label_encoder=False, eval_metric="logloss",
                random_state=42, tree_method="hist", verbosity=0,
            )
        return models

    def _ks_statistic(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob}).sort_values("y_prob", ascending=False)
        n_pos = y_true.sum()
        n_neg = len(y_true) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.0
        df["cum_pos"] = (df["y_true"] == 1).cumsum() / n_pos
        df["cum_neg"] = (df["y_true"] == 0).cumsum() / n_neg
        return float((df["cum_pos"] - df["cum_neg"]).abs().max())

    def fit(self, df: pd.DataFrame, test_size: float = 0.2) -> pd.DataFrame:
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required.")
        df = self.engineer.transform(df)
        num_cols = [c for c in self.numeric_features if c in df.columns]
        cat_cols = [c for c in self.categorical_features if c in df.columns]
        df_clean = df[num_cols + cat_cols + [self.target_col]].dropna(subset=[self.target_col])
        for col in num_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        for col in cat_cols:
            df_clean[col] = df_clean[col].fillna("unknown")

        X = df_clean[num_cols + cat_cols]
        y = df_clean[self.target_col].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        prep = self._preprocessor()
        self.results = []
        for name, est in self._estimators().items():
            pipe = Pipeline([("preprocessor", prep), ("model", est)])
            pipe.fit(X_train, y_train)
            y_prob = pipe.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            auc = float(roc_auc_score(y_test, y_prob))
            ks = self._ks_statistic(y_test.values, y_prob)
            gini = 2 * auc - 1
            report = classification_report(y_test, y_pred, output_dict=True)
            churn_f1 = report.get("1", {}).get("f1-score", 0.0)
            self.models[name] = pipe
            self.results.append({
                "model": name,
                "auc": round(auc, 4),
                "ks": round(ks, 4),
                "gini": round(gini, 4),
                "churn_f1": round(churn_f1, 4),
            })

        results_df = pd.DataFrame(self.results).sort_values("auc", ascending=False).reset_index(drop=True)
        self.best_model_name = results_df.iloc[0]["model"]
        return results_df

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.best_model_name not in self.models:
            raise RuntimeError("Call fit() first.")
        df = self.engineer.transform(df)
        num_cols = [c for c in self.numeric_features if c in df.columns]
        cat_cols = [c for c in self.categorical_features if c in df.columns]
        return self.models[self.best_model_name].predict_proba(
            df[num_cols + cat_cols]
        )[:, 1]

    def risk_segment(self, churn_prob: float) -> str:
        if churn_prob >= 0.75:
            return "critical"
        if churn_prob >= 0.50:
            return "high"
        if churn_prob >= 0.25:
            return "medium"
        return "low"

    def churn_report(self, df: pd.DataFrame) -> pd.DataFrame:
        probs = self.predict_proba(df)
        report_df = df.copy().reset_index(drop=True)
        report_df["churn_probability"] = np.round(probs, 4)
        report_df["risk_segment"] = [self.risk_segment(p) for p in probs]
        return report_df

    def feature_importance(self) -> Optional[pd.DataFrame]:
        if self.best_model_name not in self.models:
            return None
        pipe = self.models[self.best_model_name]
        est = pipe.named_steps["model"]
        if not hasattr(est, "feature_importances_"):
            return None
        prep = pipe.named_steps["preprocessor"]
        try:
            cat_names = list(prep.named_transformers_["cat"].get_feature_names_out(self.categorical_features))
        except Exception:
            cat_names = []
        names = self.numeric_features + cat_names
        imp = est.feature_importances_
        return pd.DataFrame({
            "feature": names[:len(imp)],
            "importance": imp,
        }).sort_values("importance", ascending=False).head(15).reset_index(drop=True)

    def intervention_candidates(self, df: pd.DataFrame,
                                 threshold: float = 0.5) -> pd.DataFrame:
        report = self.churn_report(df)
        return report[report["churn_probability"] >= threshold].sort_values(
            "churn_probability", ascending=False
        )


if __name__ == "__main__":
    np.random.seed(42)
    n = 3000

    df = pd.DataFrame({
        "account_length": np.random.randint(1, 250, n),
        "number_vmail_messages": np.random.randint(0, 50, n),
        "total_day_minutes": np.random.uniform(0, 350, n),
        "total_day_calls": np.random.randint(0, 200, n),
        "total_day_charge": np.random.uniform(0, 60, n),
        "total_night_minutes": np.random.uniform(0, 300, n),
        "total_night_calls": np.random.randint(0, 200, n),
        "total_night_charge": np.random.uniform(0, 25, n),
        "customer_service_calls": np.random.randint(0, 10, n),
        "international_plan": np.random.choice(["yes", "no"], n),
        "voice_mail_plan": np.random.choice(["yes", "no"], n),
        "state": np.random.choice(["MH", "DL", "KA", "TN", "UP", "WB"], n),
        "churn": np.random.binomial(1, 0.15, n),
    })

    model = TelecomChurnModel(
        numeric_features=[
            "account_length", "number_vmail_messages",
            "total_day_minutes", "total_day_calls", "total_day_charge",
            "total_night_minutes", "total_night_calls", "total_night_charge",
            "customer_service_calls",
        ],
        categorical_features=["international_plan", "voice_mail_plan", "state"],
    )

    results = model.fit(df)
    print("Model comparison:")
    print(results.to_string(index=False))
    print(f"\nBest model: {model.best_model_name}")

    report = model.churn_report(df.head(20))
    print("\nChurn risk report (first 20):")
    print(report[["account_length", "churn_probability", "risk_segment"]].to_string(index=False))

    candidates = model.intervention_candidates(df, threshold=0.5)
    print(f"\nHigh-risk customers (prob >= 0.5): {len(candidates)}")

    fi = model.feature_importance()
    if fi is not None:
        print("\nTop 5 features:")
        print(fi.head(5).to_string(index=False))
