<?php
/**
 * AI Detection Scoring Script
 * 
 * This PHP script loads a pre-trained model (exported from Python) and scores
 * text data to detect AI-generated content vs human Transcripts.
 * 
 * Usage:
 *   php score_ai_detection.php <input_csv> <output_csv> <model_json>
 * 
 * Example:
 *   php score_ai_detection.php Hold_Dec.csv Cross_Validation.csv model_for_php.json
 * 
 * Input CSV must have columns: QA (text), Type (label)
 * Output CSV will have: original columns + prob_ai, pred_label, pred_class_name, is_correct, rationale, detailed_rationale
 */

// ============================================================================
// CONFIGURATION
// ============================================================================

$TEXT_COL = "QA";
$TYPE_COL = "Type";

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Load the model from JSON file
 */
function loadModel($modelPath) {
    if (!file_exists($modelPath)) {
        die("Error: Model file not found: $modelPath\n");
    }
    
    $json = file_get_contents($modelPath);
    $model = json_decode($json, true);
    
    if ($model === null) {
        die("Error: Failed to parse model JSON\n");
    }
    
    echo "Loaded model with " . count($model['vocabulary']) . " vocabulary terms\n";
    echo "Threshold: " . $model['threshold'] . "\n";
    
    return $model;
}

/**
 * Load CSV file into array
 */
function loadCSV($csvPath) {
    if (!file_exists($csvPath)) {
        die("Error: CSV file not found: $csvPath\n");
    }
    
    $rows = [];
    $header = null;
    
    if (($handle = fopen($csvPath, "r")) !== FALSE) {
        while (($data = fgetcsv($handle, 0, ",")) !== FALSE) {
            if ($header === null) {
                $header = $data;
            } else {
                $row = [];
                for ($i = 0; $i < count($header); $i++) {
                    $row[$header[$i]] = isset($data[$i]) ? $data[$i] : "";
                }
                $rows[] = $row;
            }
        }
        fclose($handle);
    }
    
    echo "Loaded " . count($rows) . " rows from $csvPath\n";
    return $rows;
}

/**
 * Generate n-grams from tokens
 */
function generateNgrams($tokens, $minN, $maxN) {
    $ngrams = [];
    $numTokens = count($tokens);
    
    for ($n = $minN; $n <= $maxN; $n++) {
        for ($i = 0; $i <= $numTokens - $n; $i++) {
            $ngram = implode(" ", array_slice($tokens, $i, $n));
            $ngrams[] = $ngram;
        }
    }
    
    return $ngrams;
}

/**
 * Tokenize text (simple whitespace + lowercase)
 */
function tokenize($text) {
    $text = mb_strtolower($text, 'UTF-8');
    // Remove accents/special chars
    $text = preg_replace('/[^\p{L}\p{N}\s]/u', ' ', $text);
    // Split on whitespace
    $tokens = preg_split('/\s+/', $text, -1, PREG_SPLIT_NO_EMPTY);
    return $tokens;
}

/**
 * Calculate TF-IDF features for a single text
 */
function calculateTfidf($text, $vocabulary, $idfValues) {
    $tokens = tokenize($text);
    $ngrams = generateNgrams($tokens, 1, 3);  // 1-3 grams
    
    // Count term frequencies
    $termCounts = array_count_values($ngrams);
    $totalTerms = count($ngrams);
    
    // Calculate TF-IDF for each term in vocabulary
    $numFeatures = count($idfValues);
    $tfidf = array_fill(0, $numFeatures, 0.0);
    
    foreach ($termCounts as $term => $count) {
        if (isset($vocabulary[$term])) {
            $idx = $vocabulary[$term];
            // Sublinear TF: log(1 + count)
            $tf = log(1 + $count);
            $idf = $idfValues[$idx];
            $tfidf[$idx] = $tf * $idf;
        }
    }
    
    // L2 normalize
    $norm = 0.0;
    foreach ($tfidf as $val) {
        $norm += $val * $val;
    }
    $norm = sqrt($norm);
    
    if ($norm > 0) {
        for ($i = 0; $i < $numFeatures; $i++) {
            $tfidf[$i] /= $norm;
        }
    }
    
    return $tfidf;
}

/**
 * Calculate text statistics features
 * REDUCED VERSION - removed potential transcription artifacts:
 *   punct_ratio, comma_ratio, caps_ratio, digit_ratio, newline_count, space_ratio
 * Now returns 19 features (was 25)
 */
function calculateTextStats($text) {
    if (empty($text)) {
        return array_fill(0, 19, 0.0);
    }
    
    // Basic counts
    $charCount = mb_strlen($text, 'UTF-8');
    $words = preg_split('/\s+/', $text, -1, PREG_SPLIT_NO_EMPTY);
    $wordCount = count($words);
    
    // Sentences
    $sentences = preg_split('/[.!?]+/', $text, -1, PREG_SPLIT_NO_EMPTY);
    $sentences = array_filter(array_map('trim', $sentences));
    $sentenceCount = max(count($sentences), 1);
    
    // Sentence lengths
    $sentenceLengths = [];
    foreach ($sentences as $s) {
        $sentenceLengths[] = count(preg_split('/\s+/', trim($s), -1, PREG_SPLIT_NO_EMPTY));
    }
    if (empty($sentenceLengths)) $sentenceLengths = [0];
    
    $avgSentenceLength = array_sum($sentenceLengths) / count($sentenceLengths);
    $stdSentenceLength = calculateStd($sentenceLengths);
    $minSentenceLength = min($sentenceLengths);
    $maxSentenceLength = max($sentenceLengths);
    
    // Word lengths
    $wordLengths = array_map('mb_strlen', $words);
    if (empty($wordLengths)) $wordLengths = [0];
    
    $avgWordLength = array_sum($wordLengths) / count($wordLengths);
    $stdWordLength = calculateStd($wordLengths);
    
    // Long/short word ratios
    $longWordRatio = $wordCount > 0 ? count(array_filter($wordLengths, fn($l) => $l > 8)) / $wordCount : 0;
    $shortWordRatio = $wordCount > 0 ? count(array_filter($wordLengths, fn($l) => $l <= 3)) / $wordCount : 0;
    
    // Vocabulary
    $wordsLower = array_map('mb_strtolower', $words);
    $uniqueWords = count(array_unique($wordsLower));
    $vocabRichness = $wordCount > 0 ? $uniqueWords / $wordCount : 0;
    
    // Hapax ratio
    $wordFreq = array_count_values($wordsLower);
    $hapaxCount = count(array_filter($wordFreq, fn($c) => $c == 1));
    $hapaxRatio = $uniqueWords > 0 ? $hapaxCount / $uniqueWords : 0;
    
    // Keep question and exclamation counts (may indicate tone/style)
    $questionCount = substr_count($text, '?');
    $exclaimCount = substr_count($text, '!');
    
    // REMOVED: punct_ratio, comma_ratio, caps_ratio, digit_ratio, newline_count, space_ratio
    
    $wordsPerSentence = $sentenceCount > 0 ? $wordCount / $sentenceCount : 0;
    $charsPerWord = $wordCount > 0 ? $charCount / $wordCount : 0;
    
    return [
        log(1 + $charCount),           // 1
        log(1 + $wordCount),           // 2
        log(1 + $sentenceCount),       // 3
        $avgSentenceLength,            // 4
        $stdSentenceLength,            // 5
        $minSentenceLength,            // 6
        $maxSentenceLength,            // 7
        $avgWordLength,                // 8
        $stdWordLength,                // 9
        $longWordRatio,                // 10
        $shortWordRatio,               // 11
        $vocabRichness,                // 12
        $hapaxRatio,                   // 13
        $questionCount,                // 14
        $exclaimCount,                 // 15
        $wordsPerSentence,             // 16
        $charsPerWord,                 // 17
        log(1 + $uniqueWords),         // 18
        $maxSentenceLength - $minSentenceLength,  // 19
    ];
}

/**
 * Calculate standard deviation
 */
function calculateStd($values) {
    $n = count($values);
    if ($n <= 1) return 0.0;
    
    $mean = array_sum($values) / $n;
    $sumSq = 0.0;
    foreach ($values as $v) {
        $sumSq += ($v - $mean) * ($v - $mean);
    }
    return sqrt($sumSq / $n);
}

/**
 * Scale features using saved scaler parameters
 */
function scaleFeatures($features, $mean, $scale) {
    $scaled = [];
    for ($i = 0; $i < count($features); $i++) {
        $scaled[] = ($features[$i] - $mean[$i]) / $scale[$i];
    }
    return $scaled;
}

/**
 * Sigmoid function
 */
function sigmoid($x) {
    return 1.0 / (1.0 + exp(-$x));
}

/**
 * Predict probability for a single text and return features for rationale
 */
function predictProbaWithFeatures($text, $model) {
    // Calculate TF-IDF features
    $tfidf = calculateTfidf($text, $model['vocabulary'], $model['idf_values']);
    
    // Calculate text statistics features
    $stats = calculateTextStats($text);
    
    // Scale statistics features
    $statsScaled = scaleFeatures($stats, $model['scaler_mean'], $model['scaler_scale']);
    
    // Combine features
    $features = array_merge($tfidf, $statsScaled);
    
    // Calculate linear combination
    $z = $model['intercept'];
    for ($i = 0; $i < count($features); $i++) {
        $z += $features[$i] * $model['coefficients'][$i];
    }
    
    // Apply sigmoid
    $proba = sigmoid($z);
    
    return ['proba' => $proba, 'features' => $features];
}

/**
 * Generate rationale explaining the prediction
 */
function generateRationale($features, $model, $predLabel, $topN = 5) {
    $coefficients = $model['coefficients'];
    $vocabulary = $model['vocabulary'];
    $textStatNames = $model['text_stat_feature_names'];
    $nTfidf = $model['n_tfidf_features'];
    
    // Build reverse vocabulary (index -> word)
    $reverseVocab = array_flip($vocabulary);
    
    // Calculate contributions for each feature
    $contributions = [];
    for ($i = 0; $i < count($features); $i++) {
        $contrib = $features[$i] * $coefficients[$i];
        if ($contrib != 0) {
            // Get feature name
            if ($i < $nTfidf) {
                $name = isset($reverseVocab[$i]) ? $reverseVocab[$i] : "feature_$i";
            } else {
                $statIdx = $i - $nTfidf;
                $name = isset($textStatNames[$statIdx]) ? $textStatNames[$statIdx] : "stat_$statIdx";
            }
            $contributions[] = ['name' => $name, 'contrib' => $contrib];
        }
    }
    
    // Sort by absolute contribution
    usort($contributions, function($a, $b) {
        return abs($b['contrib']) <=> abs($a['contrib']);
    });
    
    // Generate rationale based on prediction
    if ($predLabel == 1) {
        // Predicted AI - show top positive contributors
        $aiFeatures = array_filter($contributions, fn($c) => $c['contrib'] > 0);
        $aiFeatures = array_slice($aiFeatures, 0, $topN);
        if (!empty($aiFeatures)) {
            $parts = array_map(fn($c) => $c['name'] . "(+" . round($c['contrib'], 2) . ")", $aiFeatures);
            return "AI indicators: " . implode(", ", $parts);
        }
        return "AI (weak signal)";
    } else {
        // Predicted Transcript - show top negative contributors
        $transFeatures = array_filter($contributions, fn($c) => $c['contrib'] < 0);
        $transFeatures = array_slice($transFeatures, 0, $topN);
        if (!empty($transFeatures)) {
            $parts = array_map(fn($c) => $c['name'] . "(" . round($c['contrib'], 2) . ")", $transFeatures);
            return "Transcript indicators: " . implode(", ", $parts);
        }
        return "Transcript (weak signal)";
    }
}

/**
 * Generate detailed rationale showing both AI and Transcript indicators
 */
function generateDetailedRationale($features, $model, $topN = 5) {
    $coefficients = $model['coefficients'];
    $vocabulary = $model['vocabulary'];
    $textStatNames = $model['text_stat_feature_names'];
    $nTfidf = $model['n_tfidf_features'];
    
    $reverseVocab = array_flip($vocabulary);
    
    $contributions = [];
    for ($i = 0; $i < count($features); $i++) {
        $contrib = $features[$i] * $coefficients[$i];
        if ($contrib != 0) {
            if ($i < $nTfidf) {
                $name = isset($reverseVocab[$i]) ? $reverseVocab[$i] : "feature_$i";
            } else {
                $statIdx = $i - $nTfidf;
                $name = isset($textStatNames[$statIdx]) ? $textStatNames[$statIdx] : "stat_$statIdx";
            }
            $contributions[] = ['name' => $name, 'contrib' => $contrib];
        }
    }
    
    usort($contributions, function($a, $b) {
        return abs($b['contrib']) <=> abs($a['contrib']);
    });
    
    $parts = [];
    
    // Top AI indicators
    $aiFeatures = array_filter($contributions, fn($c) => $c['contrib'] > 0);
    $aiFeatures = array_slice($aiFeatures, 0, $topN);
    if (!empty($aiFeatures)) {
        $aiParts = array_map(fn($c) => $c['name'] . "(+" . round($c['contrib'], 2) . ")", $aiFeatures);
        $parts[] = "AI: " . implode(", ", $aiParts);
    }
    
    // Top Transcript indicators
    $transFeatures = array_filter($contributions, fn($c) => $c['contrib'] < 0);
    $transFeatures = array_slice($transFeatures, 0, $topN);
    if (!empty($transFeatures)) {
        $transParts = array_map(fn($c) => $c['name'] . "(" . round($c['contrib'], 2) . ")", $transFeatures);
        $parts[] = "Trans: " . implode(", ", $transParts);
    }
    
    return !empty($parts) ? implode(" | ", $parts) : "No strong indicators";
}

/**
 * Save results to CSV
 */
function saveCSV($rows, $outputPath) {
    $handle = fopen($outputPath, "w");
    
    if (!empty($rows)) {
        // Write header
        fputcsv($handle, array_keys($rows[0]));
        
        // Write rows
        foreach ($rows as $row) {
            fputcsv($handle, $row);
        }
    }
    
    fclose($handle);
    echo "Saved " . count($rows) . " rows to $outputPath\n";
}

// ============================================================================
// MAIN
// ============================================================================

function main($argv) {
    global $TEXT_COL, $TYPE_COL;
    
    // Parse command line arguments
    if (count($argv) < 4) {
        echo "Usage: php score_ai_detection.php <input_csv> <output_csv> <model_json>\n";
        echo "Example: php score_ai_detection.php Hold_Dec.csv Cross_Validation.csv model_for_php.json\n";
        exit(1);
    }
    
    $inputCsv = $argv[1];
    $outputCsv = $argv[2];
    $modelJson = $argv[3];
    
    echo "\n============================================================\n";
    echo "AI DETECTION SCORING (PHP)\n";
    echo "============================================================\n\n";
    
    // Load model
    $model = loadModel($modelJson);
    $threshold = $model['threshold'];
    
    // Load input data
    $rows = loadCSV($inputCsv);
    
    // Score each row
    echo "\nScoring " . count($rows) . " rows...\n";
    
    $results = [];
    $correct = 0;
    $total = 0;
    
    foreach ($rows as $i => $row) {
        $text = isset($row[$TEXT_COL]) ? $row[$TEXT_COL] : "";
        $type = isset($row[$TYPE_COL]) ? $row[$TYPE_COL] : "";
        
        // True label: 0 = Transcript, 1 = AI
        $trueLabel = (strtolower(trim($type)) === "transcript") ? 0 : 1;
        
        // Predict with features for rationale
        $result = predictProbaWithFeatures($text, $model);
        $probAi = $result['proba'];
        $features = $result['features'];
        
        $predLabel = ($probAi >= $threshold) ? 1 : 0;
        $predClassName = ($predLabel == 1) ? "AI" : "Transcript";
        $isCorrect = ($predLabel == $trueLabel) ? 1 : 0;
        
        // Generate rationales
        $rationale = generateRationale($features, $model, $predLabel, 5);
        $detailedRationale = generateDetailedRationale($features, $model, 5);
        
        // Add prediction columns
        $row['true_label'] = $trueLabel;
        $row['prob_ai'] = round($probAi, 6);
        $row['pred_label'] = $predLabel;
        $row['pred_class_name'] = $predClassName;
        $row['is_correct'] = $isCorrect;
        $row['rationale'] = $rationale;
        $row['detailed_rationale'] = $detailedRationale;
        
        $results[] = $row;
        
        $correct += $isCorrect;
        $total++;
        
        // Progress
        if (($i + 1) % 50 == 0) {
            echo "  Processed " . ($i + 1) . " rows...\n";
        }
    }
    
    // Save results
    echo "\n";
    saveCSV($results, $outputCsv);
    
    // Print summary
    echo "\n============================================================\n";
    echo "SUMMARY\n";
    echo "============================================================\n";
    echo "Total rows: $total\n";
    echo "Correct: $correct\n";
    echo "Accuracy: " . round($correct / $total * 100, 2) . "%\n";
    echo "Threshold used: $threshold\n";
    
    // Count predictions
    $predAi = 0;
    $predTrans = 0;
    $trueAi = 0;
    $trueTrans = 0;
    $transFp = 0;  // Transcripts predicted as AI
    $aiTp = 0;     // AI correctly predicted as AI
    
    foreach ($results as $row) {
        if ($row['pred_label'] == 1) $predAi++;
        else $predTrans++;
        
        if ($row['true_label'] == 1) $trueAi++;
        else $trueTrans++;
        
        if ($row['true_label'] == 0 && $row['pred_label'] == 1) $transFp++;
        if ($row['true_label'] == 1 && $row['pred_label'] == 1) $aiTp++;
    }
    
    $transFpRate = $trueTrans > 0 ? $transFp / $trueTrans * 100 : 0;
    $aiRecall = $trueAi > 0 ? $aiTp / $trueAi * 100 : 0;
    
    echo "\nPredicted AI: $predAi\n";
    echo "Predicted Transcript: $predTrans\n";
    echo "\nTranscript FP (Transcripts wrongly flagged as AI): $transFp (" . round($transFpRate, 2) . "%)\n";
    echo "AI Recall (AI correctly detected): $aiTp / $trueAi (" . round($aiRecall, 2) . "%)\n";
    
    echo "\nDone!\n";
}

// Run
main($argv);
?>

