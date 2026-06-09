export const findHotRepos = async (topic: string): Promise<{ repos: Repo[], sources: GroundingSource[] }> => {
  const model = "gemini-2.5-flash";
  const prompt = `Find 5 currently trending or popular open source repositories related to the topic: "${topic}". 
  Focus on active projects suitable for contributors.
  
  Format Requirement:
  Return a raw JSON array of objects.
  Keys: "name", "description", "stars" (approximate count as string), "url".
  
  Do NOT include any markdown formatting or explanation text. just the JSON string.`;

  try {
    const response = await ai.models.generateContent({
      model,
      contents: prompt,
      config: {
        tools: [{ googleSearch: {} }],
      }
    });

    const text = response.text || "[]";
    const repos = extractJson(text);
    
    const chunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks || [];
    const sources: GroundingSource[] = chunks
      .map((c: any) => c.web ? { title: c.web.title, uri: c.web.uri } : null)
      .filter((s: any) => s !== null) as GroundingSource[];

    return { repos, sources };
  } catch (error) {
    console.error("Error finding repos:", error);
    return { repos: [], sources: [] };
  }
};


export const findPotentialIssues = async (repoName: string, repoUrl: string): Promise<{ issues: Issue[], sources: GroundingSource[] }> => {
    const model = "gemini-2.5-flash";
    const prompt = `Task: Find active, unassigned contribution opportunities for the repository "${repoName}" (${repoUrl}).

    Instructions:
    1. Use Google Search to find "issues" for this specific repository.
    2. STRICTLY Filter results. An issue is valid ONLY IF:
       - Status is OPEN (Explicitly ignore Closed or Merged issues).
       - It is UNASSIGNED (No assignee).
       - It has NO linked pull requests.
       - It has any labels in "good first issue", "enhancement", "help wanted", or "bug".

    Output Format:
    - Return ONLY a raw JSON array of objects.
    - Keys: "title", "url" (must be a valid url), "number" (string), "labels" (array of strings).
    - If no issues match all criteria, strictly return an empty JSON array []. 
    - Do NOT output apologies or text like "I am unable". Just [].
    `;
  
    try {
      const response = await ai.models.generateContent({
        model,
        contents: prompt,
        config: {
          tools: [{ googleSearch: {} }],
        }
      });
  
      const text = response.text || "[]";
      const issues = extractJson(text);
      
      const chunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks || [];
      const sources: GroundingSource[] = chunks
        .map((c: any) => c.web ? { title: c.web.title, uri: c.web.uri } : null)
        .filter((s: any) => s !== null) as GroundingSource[];
  
      return { issues, sources };
    } catch (error) {
      console.error("Error finding issues:", error);
      return { issues: [], sources: [] };
    }
  };


  export const analyzeIssue = async (issueTitle: string, repoName: string): Promise<IssueAnalysis | null> => {
    const model = "gemini-2.5-flash";
    const prompt = `Act as an expert Open Source Maintainer Agent. Analyze this issue titled "${issueTitle}" for the repository "${repoName}".
    Estimate the workload and difficulty.
    
    Output strictly as valid JSON:
    {
      "difficulty": (number 1-5, where 1 is easy),
      "estimated_hours": (string, e.g. "2-4 hours"),
      "required_skills": (array of strings),
      "summary": (string, brief technical explanation),
      "implementation_plan": (array of strings, step by step guide)
    }
    `;
  
    try {
      const response = await ai.models.generateContent({
        model,
        contents: prompt,
        config: {
            responseMimeType: "application/json"
        }
      });
  
      const text = response.text || "{}";
      const analysis = extractObjectJson(text);
      return analysis;
    } catch (error) {
      console.error("Error analyzing issue:", error);
      return null;
    }
  };

