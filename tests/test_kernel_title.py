import job_common as jc


def test_dataset_display_name_from_hyphenated_slug():
    assert jc.dataset_display_name("yasserh/housing-prices-dataset") == "Housing Prices Dataset"


def test_dataset_display_name_from_underscored_slug():
    assert jc.dataset_display_name("someowner/red_wine_quality") == "Red Wine Quality"


def test_dataset_display_name_handles_mixed_separators():
    assert jc.dataset_display_name("owner/some-mixed_name-here") == "Some Mixed Name Here"


def test_format_kernel_title_regression_metric_shown_as_is():
    title = jc.format_kernel_title("yasserh/housing-prices-dataset", "rmse", 1050295.6417)
    assert title == "Housing Prices Dataset (RMSE 1050295.64)"


def test_format_kernel_title_negated_metric_sign_flipped_back():
    # internal score for roc_auc is stored negative (lower-is-better convention) -
    # the displayed title must show the natural positive AUC value, not -0.93
    title = jc.format_kernel_title("owner/some-dataset", "roc_auc", -0.9321)
    assert title == "Some Dataset (ROC AUC 0.93)"


def test_format_kernel_title_f1_and_accuracy_also_flipped():
    assert jc.format_kernel_title("owner/ds", "f1", -0.812) == "Ds (F1 0.81)"
    assert jc.format_kernel_title("owner/ds", "accuracy", -0.999) == "Ds (Accuracy 1.00)"


def test_format_kernel_title_log_loss_not_flipped():
    # log_loss is naturally lower-is-better already, never negated internally
    title = jc.format_kernel_title("owner/ds", "log_loss", 0.234)
    assert title == "Ds (Log Loss 0.23)"
