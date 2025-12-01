import { NextRequest, NextResponse } from 'next/server';

const ML_API_URL = process.env.ML_API_URL || 'http://localhost:5000';

export async function POST(request: NextRequest) {
    try {
        const data = await request.json();

        // Forward request to Python ML service
        const response = await fetch(`${ML_API_URL}/predict/batch`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        if (!response.ok) {
            throw new Error('ML service error');
        }

        const result = await response.json();
        return NextResponse.json(result);
    } catch (error) {
        console.error('Batch prediction error:', error);
        return NextResponse.json(
            { error: 'Failed to get batch predictions' },
            { status: 500 }
        );
    }
}
