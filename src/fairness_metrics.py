"""Fairness metrics calculation for healthcare ML."""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score


class FairnessEvaluator:
    """Calculate fairness metrics across demographic groups."""
    
    def __init__(self, y_true, y_pred, y_prob, sensitive_attr):
        """
        Initialize fairness evaluator.
        
        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels
            y_prob (array): Predicted probabilities
            sensitive_attr (array): Sensitive attribute (e.g., skin_tone)
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_prob = np.array(y_prob)
        self.sensitive_attr = np.array(sensitive_attr)
        self.groups = np.unique(sensitive_attr)
    
    def demographic_parity_difference(self):
        """
        Calculate demographic parity difference.
        Measures difference in positive prediction rates between groups.
        """
        group_rates = {}
        
        for group in self.groups:
            mask = self.sensitive_attr == group
            positive_rate = self.y_pred[mask].mean()
            group_rates[group] = positive_rate
        
        max_rate = max(group_rates.values())
        min_rate = min(group_rates.values())
        
        return {
            'metric': 'Demographic Parity Difference',
            'value': max_rate - min_rate,
            'group_rates': group_rates,
            'interpretation': 'Lower is more fair (0 = perfect fairness)'
        }
    
    def equal_opportunity_difference(self):
        """
        Calculate equal opportunity difference.
        Measures difference in true positive rates (TPR) between groups.
        """
        group_tpr = {}
        
        for group in self.groups:
            mask = self.sensitive_attr == group
            y_true_group = self.y_true[mask]
            y_pred_group = self.y_pred[mask]
            
            # TPR = TP / (TP + FN)
            if y_true_group.sum() > 0:  # Avoid division by zero
                tpr = ((y_true_group == 1) & (y_pred_group == 1)).sum() / y_true_group.sum()
                group_tpr[group] = tpr
            else:
                group_tpr[group] = np.nan
        
        valid_tprs = [v for v in group_tpr.values() if not np.isnan(v)]
        
        if len(valid_tprs) > 1:
            diff = max(valid_tprs) - min(valid_tprs)
        else:
            diff = np.nan
        
        return {
            'metric': 'Equal Opportunity Difference',
            'value': diff,
            'group_tpr': group_tpr,
            'interpretation': 'Lower is more fair (0 = perfect fairness)'
        }
    
    def equalized_odds_difference(self):
        """
        Calculate equalized odds difference.
        Measures difference in both TPR and FPR between groups.
        """
        group_metrics = {}
        
        for group in self.groups:
            mask = self.sensitive_attr == group
            y_true_group = self.y_true[mask]
            y_pred_group = self.y_pred[mask]
            
            tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group, labels=[0, 1]).ravel()
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
            
            group_metrics[group] = {'TPR': tpr, 'FPR': fpr}
        
        # Calculate max differences
        tprs = [m['TPR'] for m in group_metrics.values() if not np.isnan(m['TPR'])]
        fprs = [m['FPR'] for m in group_metrics.values() if not np.isnan(m['FPR'])]
        
        tpr_diff = max(tprs) - min(tprs) if len(tprs) > 1 else np.nan
        fpr_diff = max(fprs) - min(fprs) if len(fprs) > 1 else np.nan
        
        return {
            'metric': 'Equalized Odds Difference',
            'tpr_difference': tpr_diff,
            'fpr_difference': fpr_diff,
            'average_difference': (tpr_diff + fpr_diff) / 2 if not np.isnan(tpr_diff) else np.nan,
            'group_metrics': group_metrics,
            'interpretation': 'Lower is more fair (0 = perfect fairness)'
        }
    
    def performance_by_group(self):
        """Calculate performance metrics for each group."""
        group_performance = {}
        
        for group in self.groups:
            mask = self.sensitive_attr == group
            y_true_group = self.y_true[mask]
            y_pred_group = self.y_pred[mask]
            y_prob_group = self.y_prob[mask]
            
            # Calculate metrics
            acc = accuracy_score(y_true_group, y_pred_group)
            
            # AUC (only if both classes present)
            if len(np.unique(y_true_group)) > 1:
                auc = roc_auc_score(y_true_group, y_prob_group)
            else:
                auc = np.nan
            
            # Confusion matrix metrics
            tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group, labels=[0, 1]).ravel()
            
            group_performance[group] = {
                'sample_size': len(y_true_group),
                'accuracy': acc,
                'auc_roc': auc,
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn),
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
            }
        
        return group_performance
    
    def generate_fairness_report(self):
        """Generate comprehensive fairness report."""
        report = {
            'demographic_parity': self.demographic_parity_difference(),
            'equal_opportunity': self.equal_opportunity_difference(),
            'equalized_odds': self.equalized_odds_difference(),
            'performance_by_group': self.performance_by_group()
        }
        
        return report
    
    def print_report(self):
        """Print formatted fairness report."""
        report = self.generate_fairness_report()
        
        print("\n" + "="*60)
        print("FAIRNESS EVALUATION REPORT")
        print("="*60)
        
        print("\n1. DEMOGRAPHIC PARITY")
        print(f"   Difference: {report['demographic_parity']['value']:.4f}")
        print("   Positive Prediction Rates by Group:")
        for group, rate in report['demographic_parity']['group_rates'].items():
            print(f"      {group}: {rate:.4f}")
        
        print("\n2. EQUAL OPPORTUNITY")
        print(f"   TPR Difference: {report['equal_opportunity']['value']:.4f}")
        print("   True Positive Rates by Group:")
        for group, tpr in report['equal_opportunity']['group_tpr'].items():
            print(f"      {group}: {tpr:.4f}")
        
        print("\n3. EQUALIZED ODDS")
        print(f"   Average Difference: {report['equalized_odds']['average_difference']:.4f}")
        print(f"   TPR Difference: {report['equalized_odds']['tpr_difference']:.4f}")
        print(f"   FPR Difference: {report['equalized_odds']['fpr_difference']:.4f}")
        
        print("\n4. PERFORMANCE BY GROUP")
        for group, metrics in report['performance_by_group'].items():
            print(f"\n   Group: {group}")
            print(f"      Sample Size: {metrics['sample_size']}")
            print(f"      Accuracy: {metrics['accuracy']:.4f}")
            print(f"      AUC-ROC: {metrics['auc_roc']:.4f}")
            print(f"      Sensitivity: {metrics['sensitivity']:.4f}")
            print(f"      Specificity: {metrics['specificity']:.4f}")
        
        print("\n" + "="*60)
