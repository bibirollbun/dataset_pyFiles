portfolio = {
    "Equity": 70000,
    "Debt": 30000
}
target_alloc = {
    "Equity": 60.0,
    "Debt": 40.0
}



class DriftDetectorAgent:
    def run(self, portfolio, target_alloc):
        total = sum(portfolio.values())
        current_pct = {k: (v / total) * 100 for k, v in portfolio.items()}
        drift = {k: current_pct.get(k, 0) - target_alloc.get(k, 0)
                 for k in target_alloc.keys()}
        explanation = []
        for k in target_alloc.keys():
            explanation.append(
                f"{k}: current {current_pct.get(k, 0):.1f}%, "
                f"target {target_alloc.get(k, 0):.1f}%, "
                f"drift {drift.get(k, 0):.1f}%"
            )
        return {
            "current_pct": current_pct,
            "drift": drift,
            "text": "\n".join(explanation)
        }


class TradePlannerAgent:
    def run(self, portfolio, target_alloc):
        total = sum(portfolio.values())
        target_values = {k: (p / 100) * total for k, p in target_alloc.items()}
        trades = {}
        for k in target_alloc.keys():
            current_val = portfolio.get(k, 0)
            diff = target_values[k] - current_val
            trades[k] = diff  # +ve = buy, -ve = sell
        return {
            "target_values": target_values,
            "trades": trades
        }


class CoachAgent:
    def run(self, drift_info, trade_plan):
        lines = []
        lines.append("Here is a simple explanation of your portfolio rebalancing:")
        lines.append("")
        lines.append("1) Current vs target allocation:")
        lines.append(drift_info["text"])
        lines.append("")
        lines.append("2) Suggested educational moves (approximate):")
        for k, diff in trade_plan["trades"].items():
            if diff > 0:
                lines.append(f"- Increase {k} by about {diff:.2f} units.")
            elif diff < 0:
                lines.append(f"- Decrease {k} by about {abs(diff):.2f} units.")
            else:
                lines.append(f"- {k} is already close to target.")
        lines.append("")
        lines.append("These are educational examples only, not real trading advice.")
        return "\n".join(lines)


class PortfolioRebalancingMentor:
    def __init__(self):
        self.drift_agent = DriftDetectorAgent()
        self.trade_agent = TradePlannerAgent()
        self.coach_agent = CoachAgent()

    def run(self, portfolio, target_alloc):
        drift_info = self.drift_agent.run(portfolio, target_alloc)
        trade_plan = self.trade_agent.run(portfolio, target_alloc)
        explanation = self.coach_agent.run(drift_info, trade_plan)
        return {
            "drift_info": drift_info,
            "trade_plan": trade_plan,
            "explanation": explanation
        }



mentor = PortfolioRebalancingMentor()
output = mentor.run(portfolio, target_alloc)
print(output["explanation"])


