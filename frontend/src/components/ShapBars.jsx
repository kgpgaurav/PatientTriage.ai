export default function ShapBars({ explanation }) {
  if (!explanation || !explanation.length) return null;
  const maxAbs = Math.max(...explanation.map((x) => Math.abs(x.contribution)));

  return (
    <div className="shap-list">
      {explanation.map((item) => {
        const pct = Math.round((Math.abs(item.contribution) / maxAbs) * 100);
        const positive = item.contribution >= 0;
        const color = positive ? "var(--band1)" : "var(--band5)";
        const left = positive ? 50 : 50 - pct / 2;
        return (
          <div className="shap-row" key={item.feature}>
            <div className="shap-name">{item.feature}</div>
            <div className="shap-bar-track">
              <div className="shap-bar" style={{ left: `${left}%`, width: `${pct / 2}%`, background: color }} />
            </div>
            <div style={{ width: 55, textAlign: "right", color }}>
              {positive ? "+" : ""}
              {item.contribution.toFixed(2)}
            </div>
          </div>
        );
      })}
    </div>
  );
}
