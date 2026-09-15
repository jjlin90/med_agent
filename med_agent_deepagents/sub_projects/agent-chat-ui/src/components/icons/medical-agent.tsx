export function MedicalAgentLogoSVG({
  className,
  width,
  height,
}: {
  width?: number;
  height?: number;
  className?: string;
}) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 64 64"
      width={width ?? 32}
      height={height ?? 32}
      className={className}
      aria-hidden="true"
    >
      <rect
        width="64"
        height="64"
        rx="18"
        fill="#0f766e"
      />
      <path
        d="M17 15h30a7 7 0 0 1 7 7v18a7 7 0 0 1-7 7H31l-10 8v-8h-4a7 7 0 0 1-7-7V22a7 7 0 0 1 7-7Z"
        fill="#fff"
      />
      <path
        d="M18 32h8l4-8 6 16 4-8h7"
        fill="none"
        stroke="#0f766e"
        strokeWidth="4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
