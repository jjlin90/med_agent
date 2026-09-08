import { NextResponse } from "next/server";

export async function GET() {
  return NextResponse.json({
    API_URL: process.env.API_URL || process.env.NEXT_PUBLIC_API_URL,
    ASSISTANT_ID:
      process.env.ASSISTANT_ID || process.env.NEXT_PUBLIC_ASSISTANT_ID,
  });
}
export const dynamic = "force-dynamic"; // 确保每次请求都重新获取
