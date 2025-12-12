import { NextResponse } from 'next/server';

const ML_API_URL = process.env.ML_API_URL || 'http://localhost:5001';

export async function GET() {
    try {
        const response = await fetch(`${ML_API_URL}/health`);

        if (!response.ok) {
            throw new Error('ML service unhealthy');
        }

        const result = await response.json();
        return NextResponse.json(result);
    } catch (error) {
        return NextResponse.json(
            { status: 'unhealthy', error: 'ML service not available' },
            { status: 503 }
        );
    }
}
