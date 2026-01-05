import { useMemo } from 'react';

export interface ScopedKnowledge {
  scopes: Record<string, string[]>;
  availableScopes: string[];
}

const GLOBAL_KEY = 'GLOBAL CONTEXT';

/**
 * Parses raw text rules into a structured map of Container -> Rules.
 * 
 * UPGRADE v4: Double-Slash Block Nomenclature
 * Supports explicit container delimiters:
 * // NAME CONTAINER
 * ... rules ...
 * // END OF NAME CONTAINER
 */
export const useKnowledgeScoper = (rawRules: string | undefined): ScopedKnowledge => {
  return useMemo(() => {
    if (!rawRules) {
      return { 
        scopes: { [GLOBAL_KEY]: [] }, 
        availableScopes: [GLOBAL_KEY] 
      };
    }

    // Initialize with Global Context
    const scopes: Record<string, string[]> = { [GLOBAL_KEY]: [] };
    let currentScope = GLOBAL_KEY;

    // Split and cleanup whitespace but preserve indices for context
    const lines = rawRules.split('\n');

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line) continue;

      // --- HEURISTIC 1: Explicit Syntax Headers ---
      
      // 1A. Double-Slash Block Nomenclature (Priority)
      const startContainerMatch = line.match(/^\/\/\s*(.+?)\s+CONTAINER$/i);
      
      // Loose Termination Guard: Aggressively catch "END OF" variations to prevent ghost zones
      const endContainerMatch = line.match(/^\/\/\s*END\s+OF\s+(.+?)\s+CONTAINER$/i);
      const looseEndMatch = line.match(/^(?:\/\/|\*\*|##)?\s*(?:END\s+OF)\s+(.+?)(?:\s+CONTAINER)?(?:\*\*|##)?$/i);

      if (endContainerMatch || looseEndMatch) {
          // Explicitly terminate scope and return to global
          currentScope = GLOBAL_KEY;
          continue; // Consume the end tag line
      }

      let detectedHeader: string | null = null;

      if (startContainerMatch) {
          detectedHeader = startContainerMatch[1];
      }

      // 1B. Standard Markdown/Text Headers (Fallback)
      if (!detectedHeader) {
          const bracketMatch = line.match(/^\[([^\]]+)\]:?$/);         // [Header]
          const mdMatch = line.match(/^#+\s+(.+)$/);                   // ### Header
          const boldMatch = line.match(/^\*\*([^*]+)\*\*:?$/);         // **Header**
          const colonMatch = line.match(/^([A-Za-z0-9 _/-]+):$/);      // Header:

          if (bracketMatch) detectedHeader = bracketMatch[1];
          else if (mdMatch) detectedHeader = mdMatch[1];
          else if (boldMatch) detectedHeader = boldMatch[1];
          else if (colonMatch && line.length < 50) detectedHeader = colonMatch[1];
      }

      // --- HEURISTIC 2: Implicit "Title Case" Headers with Look-Ahead ---
      // Only apply if we are currently in Global Scope (don't break explicit blocks with loose heuristics)
      if (!detectedHeader && currentScope === GLOBAL_KEY) {
          // Check next non-empty line
          let nextIndex = i + 1;
          while (nextIndex < lines.length && !lines[nextIndex].trim()) {
              nextIndex++;
          }
          const nextLine = nextIndex < lines.length ? lines[nextIndex].trim() : '';

          // Is the NEXT line a list item? (starts with "1.", "-", "*", "•")
          const isNextList = /^(?:\d+\.|[-*•])\s/.test(nextLine);
          
          // Is THIS line NOT a list item?
          const isCurrentList = /^(?:\d+\.|[-*•])\s/.test(line);

          // Is this line short enough to be a header?
          const isShort = line.length < 60;

          // Does it NOT end in punctuation (like a sentence)?
          const isSentence = /[.!?]$/.test(line);

          if (isNextList && !isCurrentList && isShort && !isSentence) {
              detectedHeader = line;
          }
      }

      // --- APPLY SCOPING ---
      if (detectedHeader) {
          const normalizedScope = detectedHeader.trim().toUpperCase();
          
          // Ignore system tags
          if (normalizedScope !== 'START KNOWLEDGE' && normalizedScope !== 'END KNOWLEDGE') {
              currentScope = normalizedScope;
              if (!scopes[currentScope]) {
                  scopes[currentScope] = [];
              }
          }
      } else {
          // It is content. Add to current scope.
          scopes[currentScope].push(line);
      }
    }

    return {
      scopes,
      availableScopes: Object.keys(scopes)
    };
  }, [rawRules]);
};