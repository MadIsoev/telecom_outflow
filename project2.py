from sklearn.metrics import roc_auc_score, roc_curve

# Предсказания вероятностей для положительного класса
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Проверка, что y_pred_proba не пустой
if len(y_pred_proba) == 0:
    st.error("Ошибка при получении предсказаний вероятности.")
else:
    # ROC-кривая
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='red', linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC-Кривая')
    ax.legend(loc='lower right')
    st.pyplot(fig)
