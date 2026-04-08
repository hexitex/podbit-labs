export type ErrorCategory = 'llm' | 'sandbox' | 'evaluation' | 'system' | 'timeout' | 'cancelled';

export class LabError extends Error {
    readonly category: ErrorCategory;
    readonly retryable: boolean;

    constructor(message: string, category: ErrorCategory, retryable: boolean = false) {
        super(message);
        this.name = 'LabError';
        this.category = category;
        this.retryable = retryable;
    }
}

/**
 * Thrown when an LLM response is truncated due to token limits.
 * Contains the partial content for potential recovery by callers.
 * Not retryable by default — retrying the same prompt will produce the same truncation.
 */
export class TruncatedError extends LabError {
    readonly partialContent: string;

    constructor(partialContent: string) {
        super('LLM response truncated (token limit reached)', 'llm', false);
        this.name = 'TruncatedError';
        this.partialContent = partialContent;
    }
}
