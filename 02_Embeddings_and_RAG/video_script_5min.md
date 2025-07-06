# Video Script: Comprehensive Analysis of Distance Metrics in RAG Systems
## Duration: 5 minutes | ~750 words at 150 words/minute

---

## [0:00-0:20] Opening Hook & Introduction

**[VISUAL: Title slide with "Which Distance Metric Makes RAG Systems Smarter?"]**

"What if I told you that the way your RAG system measures similarity could increase hallucinations by 36%? Or that the most popular distance metric might be the worst choice for your application?

Hi, I'm [Your Name], and today I'll share groundbreaking findings from my comprehensive analysis of distance metrics in Retrieval-Augmented Generation systems. This isn't just theory – it's based on rigorous testing with real data and statistical validation."

---

## [0:20-1:00] The Problem & Research Question

**[VISUAL: RAG architecture diagram highlighting the retrieval component]**

"RAG systems have revolutionized how we build AI applications, but there's a critical component that's often overlooked: the distance metric used in vector retrieval.

Most developers default to cosine similarity without questioning if it's optimal. But what if different metrics could dramatically improve performance?

I set out to answer: **Which distance metric produces the most accurate, reliable, and safe RAG responses?**

To find out, I implemented and tested four major distance metrics:
- Cosine Similarity – the industry standard
- Euclidean Distance – straight-line measurement
- Manhattan Distance – city-block distance
- Dot Product – magnitude-sensitive similarity"

---

## [1:00-1:45] Methodology & Implementation

**[VISUAL: Code snippet showing the distance metric implementations]**

"Here's what makes this analysis unique. First, I implemented all four distance metrics from scratch, ensuring consistent ranking where higher scores mean better matches.

**[VISUAL: Evaluation framework diagram]**

Then, I built a comprehensive evaluation framework measuring six dimensions:
1. Context Relevance – how well retrieved documents match queries
2. Answer Faithfulness – adherence to retrieved context
3. Answer Relevance – how well answers address questions
4. Context Utilization – percentage of context used
5. Hallucination Risk – unsupported claims and false confidence
6. Response Length – verbosity analysis

**[VISUAL: Dataset statistics]**

I tested on the PMarca blog corpus with 373 document chunks, using OpenAI's text-embedding-3-small model and GPT-4o-mini for generation. Five diverse queries tested different retrieval scenarios."

---

## [1:45-2:45] Key Findings & Surprising Results

**[VISUAL: Performance ranking bar chart]**

"The results challenged everything I expected. Manhattan Distance emerged as the clear winner with a composite score of 0.729, but that's not the whole story.

**[VISUAL: Heatmap showing all metrics with identical context relevance]**

Here's the first shock: ALL distance metrics showed identical context relevance at 0.333. This means the embedding model quality completely dominates retrieval – the distance calculation method doesn't affect which documents get selected!

**[VISUAL: Scatter plot of faithfulness vs hallucination risk]**

But here's where it gets fascinating. I discovered a paradox: the most faithful responses have the HIGHEST hallucination risk. Look at this correlation of 0.72 between faithfulness and hallucinations.

Dot Product achieves 98.3% faithfulness but has a hallucination risk of 3.0. Manhattan has 92.7% faithfulness but only 2.2 hallucination risk. That's a 27% reduction in hallucinations for just a 6% faithfulness trade-off!

**[VISUAL: Radar chart comparing all metrics]**

This radar chart reveals each metric's personality:
- Manhattan: The balanced performer
- Dot Product: Faithful but risky
- Euclidean: The verbose middle ground  
- Cosine: Consistently mediocre"

---

## [2:45-3:45] Statistical Validation & Trade-off Analysis

**[VISUAL: ANOVA results table]**

"These aren't just observations – they're statistically significant. ANOVA testing confirmed significant differences in answer faithfulness (p=0.002), answer relevance (p=0.011), and hallucination risk (p<0.001).

**[VISUAL: Trade-off visualization]**

The critical trade-offs are clear:
1. **Faithfulness vs Safety**: Accept 6% less faithfulness for 27% fewer hallucinations
2. **Relevance vs Verbosity**: Manhattan achieves highest relevance with optimal 1,200-character responses
3. **Performance vs Consistency**: 4% performance gain is worth moderate variance

**[VISUAL: Use case matrix]**

This led me to create a selection framework:
- **Legal/Medical/Financial**: Use Manhattan for safety
- **Research & Analysis**: Euclidean for balance
- **Creative Writing**: Dot Product for comprehensiveness
- **Real-time Chat**: Cosine for speed"

---

## [3:45-4:30] Implementation Recommendations

**[VISUAL: Implementation checklist]**

"So how do you apply these findings? Here's your action plan:

**Immediate Actions:**
1. Replace Cosine with Manhattan Distance in production
2. Monitor hallucination risk – keep it under 2.5
3. Cap responses at 1,300 characters
4. Add confidence scoring

**[VISUAL: Hybrid approach diagram]**

For optimal results, implement a hybrid approach: Use Dot Product for initial broad retrieval of 20 documents, then re-rank with Manhattan Distance to select the final 3. This combines comprehensive coverage with precision.

**[VISUAL: Monitoring dashboard mockup]**

Set up monitoring for:
- Real-time hallucination alerts
- Answer relevance distribution
- Query-type performance breakdown"

---

## [4:30-5:00] Conclusions & Call to Action

**[VISUAL: Summary statistics and key takeaways]**

"This analysis proves that thoughtful distance metric selection can dramatically improve RAG performance. Manhattan Distance offers the best balance of accuracy, safety, and relevance for most applications.

Three key insights to remember:
1. Distance metrics don't affect retrieval quality – embeddings do
2. Higher faithfulness paradoxically increases hallucination risk
3. Manhattan Distance reduces hallucinations by 27% with minimal trade-offs

**[VISUAL: GitHub repo and contact information]**

Want to implement these findings? Check out my GitHub repository for the complete code, evaluation framework, and detailed analysis. 

This is just the beginning – imagine metric-aware embeddings, dynamic selection algorithms, and new distance metrics designed specifically for RAG.

What distance metric are you using in your RAG system? Let me know in the comments, and don't forget to star the repository if this helped you build better AI applications.

Thanks for watching!"

---

## Director's Notes:

### Pacing Guide:
- **Energy**: Start high, maintain engagement with surprising findings
- **Visuals**: Change every 15-20 seconds to maintain attention
- **Emphasis**: Stress the 27% hallucination reduction and paradox discovery
- **Tone**: Authoritative but approachable, data-driven but accessible

### Key Visual Elements:
1. Performance comparison charts
2. Code snippets (brief, highlighted)
3. Statistical results (simplified)
4. Trade-off visualizations
5. Implementation framework

### Speaking Tips:
- Pause after major revelations
- Use hand gestures for numerical comparisons
- Maintain eye contact during key findings
- Smile when revealing surprising results

Remember: Your enthusiasm for the surprising discoveries will make the technical content engaging!