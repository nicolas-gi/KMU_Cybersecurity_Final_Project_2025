import { NextResponse } from 'next/server';

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:5001';

export async function GET() {
    try {
        const response = await fetch(`${ML_SERVICE_URL}/simulate`);

        if (!response.ok) {
            throw new Error(`ML service responded with status ${response.status}`);
        }

        const data = await response.json();
        return NextResponse.json(data);
    } catch (error) {
        console.error('Error calling ML service:', error);
        return NextResponse.json(
            { error: 'Failed to fetch traffic sample' },
            { status: 500 }
        );
    }
}
