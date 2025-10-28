import streamlit as st

# Show directions/info based on model
def main(model_choice):

    metrics =  ("**Accuracy**: The proportion of all predictions that the model got correct.\n\n"
    "**Precision**: Out of all cases predicted positive (cancer), how many were actually positive? Hence, high precision indicates fewer false alarms.\n\n"
    "**Recall**: Out of all actual positives (cancer cases), how many did the model catch? Hence, higher recall indicates fewer missed cases\n\n"
    "**F1 Score**: Harmonic Mean of the Precision and Recall hence balancing the two metrics.")
    if model_choice == "Random Forest":





        st.markdown("""
###  Random Forest (Classifier)

An **ensemble** of many decision trees trained on random subsets of the data and features. Each tree is a weak learner; together they vote to make a strong, stable predictor. This is called **bagging** (bootstrap aggregating).

**How it predicts**
1) Train many trees on bootstrapped samples.  
2) At each split, a random subset of features is considered (decorrelates trees).  
3) **Classification:** each tree votes; the forest returns the **majority class**.  
   **Probabilities** are the **average** of per-tree probabilities.

**Why it works**
   - Averaging across de-correlated trees **reduces variance/overfitting** while keeping flexible, non-linear decision boundaries.



   **Strengths**
   - Handles **non-linear** relations and **feature interactions**.  
   - Works well on **tabular data** with mixed scales; minimal preprocessing.  
   - **Robust** to outliers and noisy features.  
   - Offers **feature importance** to see which inputs matter.

   """)

        st.image("rf_diagram.png", caption="Random Forest illustration")





    elif model_choice == "Decision Tree":
        
        st.markdown("""
### Decision Trees

A tree-like model with roots, nodes and branches that makes predictions by asking a sequence of if/else questions about the data. Each internal node is a condition, each branch is an outcome, and each leaf is a final prediction (class label).
**How it predicts**
1. Start at the root node.
2. Ask a question about a feature (e.g., “Is tumor size > 2 cm?”).

3. Follow the branch (yes/no or threshold comparison).

4. Repeat at the next node until you reach a leaf node.

5. The leaf gives the predicted class (e.g., benign or malignant).


                    
**Why it works**
- Splits the data into regions where classes are more “pure” (homogeneous).

- Each split reduces uncertainty (measured by criteria like Gini impurity or entropy).

- Very interpretable → rules can be read directly as a sequence of decisions.
                    

**Strengths**
- Very easy to understand and visualize.

- Handles both numeric and categorical features.

- No need to scale/normalize inputs.
                    
**Limitations**
- Prone to overfitting if not pruned or depth-limited.

- Can be unstable: small data changes can produce a different tree.

""")
        st.image("Decision-tree-1.webp", caption="Decision Tree illustration")

    elif model_choice == "SVC":
        
        st.markdown("""
### Support Vector Classifier (SVC)

A supervised learning algorithm that tries to find the **best separating boundary (hyperplane)** between classes.  
The “support vectors” are the critical data points closest to that boundary — they define where the decision line sits.

**How it predicts**
1. Finds a hyperplane that **maximizes the margin** between classes.  
2. Only the **support vectors** (borderline cases) influence this boundary.  
3. **Classification:** new samples are placed relative to the hyperplane.  
   **Probabilities** can be estimated using Platt scaling (`probability=True` in scikit-learn).

**Key settings**
- `kernel` — defines the shape of the decision boundary:  
  - `'linear'`: straight line/plane.  
  - `'rbf'`: flexible, handles non-linear patterns (most common).  
  - `'poly'`, `'sigmoid'`: other specialized kernels.  
- `C` — regularization strength:  
  - Small `C` → wider margin, more tolerance for misclassifications (simpler model).  
  - Large `C` → tighter margin, fewer errors on training data (risk of overfitting).  
- `gamma` (for RBF/poly kernels) — controls influence of single points:  
  - Low `gamma` → smoother boundary, looks far.  
  - High `gamma` → complex boundary, fits closely around points.

**Strengths**
- Effective in **high-dimensional spaces**.  
- Works well when classes are **well separated**.  
- Can model **non-linear** relationships with kernels.  
- Often robust even when number of features > number of samples.

""")
        st.image("svm.webp", caption="Support Vector Machine illustration")
