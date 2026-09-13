import { cn } from "@/lib/utils";
import React from "react";

interface ShimmerBadgeProps {
  children: React.ReactNode;
  className?: string;
  shimmerColor?: string;
}

export function ShimmerBadge({
  children,
  className,
}: ShimmerBadgeProps) {
  return (
    <div
      className={cn(
        "inline-flex items-center gap-2 rounded-full border border-white/[0.1] bg-zinc-950/80 px-3 py-1 text-xs text-zinc-300 backdrop-blur-md transition-colors hover:border-white/20",
        className
      )}
    >
      <span className="inline-block bg-[linear-gradient(110deg,#71717a,45%,#fafafa,55%,#71717a)] bg-[length:200%_100%] bg-clip-text text-transparent animate-shimmer font-mono font-medium">
        {children}
      </span>
    </div>
  );
}
