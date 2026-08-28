def build_prompt(summary, unusual_count):
    return f"""
ROLE:
You are a student budgeting assistant.
DATA:
Total Spending: ■{summary["total"]:,.2f}
Average Transaction: ■{summary["average"]:,.2f}
Largest Spending Category: {summary["top_category"]}
Share of Largest Category: {summary["top_percentage"]:.1f}%
Unusual Spending Alerts: {unusual_count}
TASK:
1. Explain the overall spending pattern in simple language.
2. Mention the largest spending category.
3. Give three practical budgeting suggestions.
4. Explain that unusual spending alerts should be checked by the user.
RULES:
- Do not invent financial figures.
- Do not call an unusual transaction fraud.
- Do not make financial decisions for the user.
- Keep the explanation simple.
""".strip()
def local_summary(summary, unusual_count):
    return (
        f"Your total sample spending is "
        f"■{summary['total']:,.2f}. "
        f"The largest spending category is "
        f"{summary['top_category']} "
        f"({summary['top_percentage']:.1f}% of total spending). "
        f"The system identified {unusual_count} unusual spending alert(s). "
        f"Consider setting spending limits for discretionary categories "
        f"and checking high-value purchases."
    )
