import clsx from "clsx";

type ConcernLevel = "low" | "moderate" | "high" | "unknown";

interface StatusBadgeProps {
  level: ConcernLevel;
  size?: "sm" | "md";
}

const config: Record<ConcernLevel, { label: string; classes: string }> = {
  low: {
    label: "Low Risk",
    classes: "bg-[#F0FDF4] text-[#16A34A] border-[#BBF7D0]",
  },
  moderate: {
    label: "Moderate Risk",
    classes: "bg-[#FFFBEB] text-[#B45309] border-[#FEF08A]",
  },
  high: {
    label: "High Risk",
    classes: "bg-[#FEF2F2] text-[#DC2626] border-[#FECACA]",
  },
  unknown: {
    label: "Undetermined",
    classes: "bg-[#F8FAFC] text-[#475569] border-[#E5EAF0]",
  },
};

export default function StatusBadge({ level, size = "md" }: StatusBadgeProps) {
  const cfg = config[level];
  return (
    <span
      className={clsx(
        "inline-flex items-center font-semibold rounded-full border",
        cfg.classes,
        size === "sm" ? "px-2.5 py-0.5 text-[10px]" : "px-3 py-1 text-xs"
      )}
    >
      <span
        className={clsx(
          "rounded-full mr-1.5",
          size === "sm" ? "w-1.5 h-1.5" : "w-2 h-2",
          level === "low"
            ? "bg-[#16A34A]"
            : level === "moderate"
            ? "bg-[#F59E0B]"
            : level === "high"
            ? "bg-[#EF4444]"
            : "bg-[#94A3B8]"
        )}
      />
      {cfg.label}
    </span>
  );
}
