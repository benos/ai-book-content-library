# AI for Designers — Research Brief: Source Entries
**Research date:** 26 April 2026  
**Compiled for:** *AI for Designers* — proof-of-concept book, first version  
**Agent note:** Entries are structured for use as generation-time source material. Interpretation is opinionated and grounded. Role-specific translation happens at book generation, not here.

---

## 1. MECHANISTIC — How Models Work

---

### How Language Models Learn: Pattern Recognition, Not Rule-Following

**Facts:**  
Large language models are neural networks trained on vast quantities of text. The core task during training is next-token prediction: given a sequence of words, predict what comes next. This is done billions of times across trillions of tokens of internet text, books, and code. The model is not given explicit rules; it learns statistical regularities — which words, ideas, and structures tend to follow which others. Models based on the transformer architecture (introduced in the 2017 "Attention Is All You Need" paper) now dominate the field. Once trained, a base model can be further aligned to human preferences through reinforcement learning from human feedback (RLHF) and instruction tuning, which shifts it from next-word predictor to usable assistant. Recent interpretability research from Anthropic (published 2025) examined Claude's internal processing and found evidence that the model genuinely reasons across intermediate conceptual steps — not merely pattern-matching surface text — and that it appears to operate via a language-agnostic conceptual space shared across languages.

**Interpretation:**  
The key insight for designers is the distinction between *statistical fluency* and *understanding*. The model produces outputs that are statistically consistent with its training distribution — which means it can sound confident and plausible even when it has no factual grounding. This is the structural cause of hallucination: the model is not retrieving facts, it is generating text that patterns well. The conceptual sophistication found in interpretability research complicates the simple "stochastic parrot" framing, but it doesn't resolve the grounding problem. A designer who understands this will know why AI copy needs fact-checking, why AI-generated UX rationale can sound coherent but be wrong, and why asking an LLM to reason about visual space (which it can't perceive) will produce confident nonsense.

**Open questions:**  
How much of LLM behaviour is genuine conceptual reasoning versus very sophisticated pattern matching? The interpretability research is promising but incomplete. Whether model scale alone eventually produces something closer to reliable factual grounding remains genuinely contested.

**Sources:**  
- Wikipedia, *Large language model*, accessed April 2026, https://en.wikipedia.org/wiki/Large_language_model  
- Anthropic, *Tracing the Thoughts of a Large Language Model*, 2025, https://www.anthropic.com/research/tracing-thoughts-language-model  
- Medium (Jacob Grow), *How Do Large Language Models Work? Conceptual But Non Technical Explanation*, May 2024, https://medium.com/@Gbgrow/how-do-large-language-models-work-conceptual-but-non-technical-explanation-ea369334d32e

---

### What a Prompt Actually Does

**Facts:**  
Prompting a language model is not issuing a command to a database or rule engine. It is providing context that steers the model's probability distribution over its next output. The model's responses are sensitive to subtle variations in wording, structure, and framing — some studies have shown accuracy variations of up to 76 percentage points from formatting changes alone. In-context learning (where a model learns from examples in the prompt itself) is an emergent property of model scale; it is temporary, producing no lasting change to model weights. A 2024 survey identified over 50 distinct text-based prompting techniques and 40 multimodal variants. The model cannot follow a prompt perfectly — it is approximating what the prompt implies, within the probability landscape its training created. Reasoning models (introduced by OpenAI in late 2024 with o1, followed by o3 in April 2025, and mirrored by DeepSeek-R1) add a step-by-step internal reasoning chain before outputting a final answer, improving performance on complex tasks.

**Interpretation:**  
For designers, the practical implication is that prompt quality functions like brief quality. A vague brief produces variable, often mediocre output. A specific brief with examples, constraints, and success criteria produces better work. The instinct to type a casual phrase and expect professional output is the same mistake as handing a junior designer a one-line brief. The non-determinism is structurally important: because outputs vary, designers should treat AI generation as ideation (produce many, select carefully) rather than execution (produce one, use it). This reframe changes the workflow significantly.

**Open questions:**  
Whether better prompting or better models will matter more in the long run is unclear — there is an argument that as models become more capable, prompting becomes less skill-dependent. The value of prompt engineering as a persistent skill is contested.

**Sources:**  
- Wikipedia, *Prompt engineering*, accessed April 2026, https://en.wikipedia.org/wiki/Prompt_engineering  
- PICCO Framework paper, accessed April 2026, https://arxiv.org/pdf/2604.14197

---

### How Image Generation Models Work: Denoising from Noise

**Facts:**  
Current leading image generation models (Stable Diffusion, Midjourney, DALL-E, Adobe Firefly, Flux) are predominantly based on diffusion models. The core mechanism has two phases: a forward process that progressively adds Gaussian noise to training images until they become pure noise, and a reverse process that trains a neural network to remove noise step by step. At inference time, the model starts from pure noise and iteratively denoises, guided by a text prompt (via a text encoder such as CLIP). The result is an image that is statistically consistent with the prompt and with the model's learned distribution of what images look like. Latent diffusion models (used in Stable Diffusion and its successors) run this process in a compressed latent space rather than full pixel space, which dramatically reduces computational cost. Research has shown that diffusion generation proceeds "outline first, details later": high-level composition is determined in early denoising steps, fine details in later ones — a structure somewhat analogous to how human painters work from broad to specific.

**Interpretation:**  
This architecture explains several things designers encounter in practice. First, outputs are generative, not retrieved: the model is not finding images from a database, it is synthesising them from a noise-to-image path that passes through its learned understanding of visual probability. This is why you cannot get "the image I have in mind" with a prompt — you can only get "an image the model considers plausible given your words." Second, variation is structural: generating with the same prompt twice produces different images because the noise seed differs. Third, text prompting for images is closer to art direction than search: you are describing aesthetic qualities, mood, composition, and style, not specifying precise visual content. The model infers the rest.

**Open questions:**  
How much do current models actually "understand" visual concepts versus recognising statistical co-occurrence between text labels and visual features? The question matters for capability assessment. Whether transformer-based architectures (increasingly replacing U-Net backbones) produce fundamentally different capabilities or just better scaling remains an open research question.

**Sources:**  
- IBM Think, *What are Diffusion Models?*, November 2025, https://www.ibm.com/think/topics/diffusion-models  
- SuperAnnotate, *Introduction to Diffusion Models for Machine Learning*, February 2025, https://www.superannotate.com/blog/diffusion-models  
- arXiv, *Diffusion Models Generate Images Like Painters*, 2023/2024, https://arxiv.org/abs/2303.02490  
- Wikipedia, *Diffusion model*, accessed April 2026, https://en.wikipedia.org/wiki/Diffusion_model

---

### The Training Data Question: What "Trained on the Internet" Actually Means

**Facts:**  
Both language and image generation models are trained on massive datasets scraped from the internet, supplemented in some cases by licensed or curated content. Midjourney's founder has described its dataset as "a big scrape of the internet," including LAION datasets (compiled from billions of URL-image pairs). Adobe Firefly is a notable exception: it is trained exclusively on licensed Adobe Stock imagery, content with expired copyright, and content Adobe owns — positioning it as the commercially "safe" option. The LAION-5B dataset, used to train several versions of Stable Diffusion, contained approximately 5.85 billion URL-text pairs from public web sources, including copyrighted images. "Trained on the internet" means the model's stylistic tendencies, biases, and knowledge gaps directly reflect what the internet contains and emphasises: overrepresentation of English-language content, Western visual aesthetics, certain demographics and body types, and the visual languages of photographers and illustrators whose work was widely shared online. Research has confirmed that image generation models exhibit statistical biases in gender, skin colour, and geography even for ostensibly neutral prompts.

**Interpretation:**  
For designers, this has two practical consequences. One is aesthetic: AI-generated images reflect the aggregate taste of mass internet imagery — technically competent, but stylistically convergent toward a specific mid-2010s photographic aesthetic. Breaking out of this default requires deliberate prompting. The other is ethical and legal: the creative work of real human designers and illustrators — often without consent — is structurally embedded in what these models can produce. This is not a bug being addressed; it is the foundation of the technology. Designers who use these tools are participating in this system, and understanding that they are is important for making conscious choices about when and how to use it.

**Open questions:**  
The legal status of training on copyrighted content remains genuinely unresolved globally (see Section 6). Whether future models trained on licensed or synthetic data will produce qualitatively different outputs is unknown — the current evidence suggests they may be somewhat less capable at matching stylistic diversity.

**Sources:**  
- Wikipedia, *Midjourney* (Holz training data quote), accessed April 2026, https://en.wikipedia.org/wiki/Midjourney  
- Adobe Firefly enterprise site, training data claims, accessed April 2026, https://business.adobe.com/products/firefly-business/firefly-ai-approach.html  
- Midjourney Wikipedia (bias research references), accessed April 2026

---

## 2. CAPABILITY MAP — What AI Can and Cannot Do

---

### What Image Generation Is Genuinely Good At

**Facts:**  
Current image generation tools excel in specific domains well-evidenced by professional use: mood-boarding and visual research (generating reference imagery quickly across many aesthetic directions); style exploration (applying different visual treatments to the same concept); variation generation (producing multiple versions of a scene or composition for comparison); ideation at volume (generating more starting points than a human could sketch in the same time); concept illustration (giving rough ideas visual form for discussion); and texture and surface pattern generation. These capabilities are genuine and well-demonstrated by practitioners.

**Interpretation:**  
The common thread across all of these is that they work *with* variation and approximation rather than against it. A mood board benefits from loose interpretation. A style exploration wants divergent outputs. A concept illustration needs to be good enough to communicate, not precise. These are the use cases where AI generation delivers value without requiring precision or consistency — the structural properties the technology does well. Designers who reframe their workflow to lean into these strengths (treating AI as an ideation engine rather than an execution engine) report genuine productivity and creative gains. The trap is using AI for tasks that require control, reproducibility, and specificity.

**Open questions:**  
How well do these benefits hold at professional creative standards versus good-enough-for-internal-discussion standards? The gap between "useful for a client presentation" and "ready to use" is still significant for most outputs, and varies considerably by use case.

**Sources:**  
- Figma, *AI in Design (2025 survey data)*, https://www.figma.com/resource-library/ai-in-design/  
- Figma 2025 AI Report, https://www.figma.com/reports/ai-2025/

---

### What Image Generation Is Genuinely Bad At

**Facts:**  
Current image generation tools have documented, structural limitations in several areas:  
- **Consistency**: Generating the same character, product, or UI element twice and having it look identical is very hard without workarounds (ControlNet, IP-Adapter, LoRA fine-tuning). This is a fundamental property of the stochastic generation process.  
- **Text rendering**: Historically, text in images was essentially unusable. Models including Midjourney V6 (Dec 2023) and Ideogram made significant progress. By 2025/26, text rendering is functional in some tools but still unreliable for precise typographic requirements.  
- **Precise spatial relationships**: Getting objects to appear at exact coordinates, in exact proportions, or in specific spatial arrangements relative to each other remains difficult.  
- **Anatomy**: Hands, feet, and complex anatomical positions remain problematic across most models, though Midjourney V7 (April 2025) made notable improvements.  
- **Brand-specific accuracy**: Producing outputs that match a specific brand's visual identity (specific product designs, logo elements, proprietary colour systems) without fine-tuning is essentially impossible. The model has no knowledge of proprietary assets.  
- **Reproducibility**: The same prompt produces different outputs on different runs. Without seed locking and careful parameter management, recreating a specific output is not guaranteed.

**Interpretation:**  
These limitations are not temporary engineering problems awaiting a software patch — several are structural properties of how diffusion models work. The consistency problem, in particular, is deeply challenging because variation is baked into the generation mechanism. Tools like ControlNet, IP-Adapter, and InstantID have made significant progress, but they introduce workflow complexity that requires technical skill. Designers who expect AI to replace finished asset production — rather than augment early-stage ideation — consistently report disappointment. The tools are considerably better than nothing for exploration, and considerably worse than human execution for precision.

**Open questions:**  
The pace of improvement on anatomy and text rendering has been faster than expected. Whether consistency will reach a level practical for brand asset production without extensive fine-tuning remains genuinely unclear in a 2–3 year window.

**Sources:**  
- Midjourney documentation and version history, accessed April 2026, https://docs.midjourney.com  
- Stable Diffusion Art, *How to create consistent character from different viewing angles*, August 2024, https://stable-diffusion-art.com/consistent-character-view-angle/  
- Skywork AI, *How to Keep Characters Consistent Across AI Scenes*, 2025, https://skywork.ai/blog/how-to-consistent-characters-ai-scenes-prompt-patterns-2025/

---

### The Consistency Problem in Detail

**Facts:**  
Generating a consistent character, product, or visual identity element across multiple images is one of the hardest problems in AI image generation. The primary solutions as of 2025/26 are:  
- **ControlNet**: Controls structural aspects of generation (pose, edges, depth) using reference conditioning, but does not preserve identity — it controls composition, not appearance.  
- **IP-Adapter**: Extracts visual style or identity features from a reference image and uses them to guide generation. IP-Adapter Face ID variants use facial landmark extraction (InsightFace) and can transfer specific face identity to new scenes with reasonable fidelity.  
- **InstantID**: Zero-shot identity preservation from a single reference image, combining identity features with facial landmarks. Reported stronger identity fidelity than CLIP-only adapters.  
- **LoRA fine-tuning**: Training a lightweight adapter on 10–30 images of a specific subject or style produces more reliable identity preservation, but requires technical setup and compute resources.  
- **Midjourney V7 (April 2025)** introduced improved character consistency natively, addressing one of the most common user complaints about earlier versions.

The dominant finding from practitioners is that achieving strong consistency typically requires combining multiple techniques (ControlNet + IP-Adapter + prompt constraints) and still involves iteration and imperfection. The workflow complexity is significant.

**Interpretation:**  
For designers, this is the critical capability gap for production use cases. Illustration projects, brand visual systems, character design for products, and any project requiring a recognisable recurring visual element all run into this wall. The tools available represent genuine progress but are not a solved problem. A designer making informed decisions needs to know: consistency for approved production assets still requires either extensive iteration time, technical pipeline investment, or both. Treating AI generation as a finished-output tool for brand-consistent visual work will produce disappointment. As an ideation tool for exploring directions, it can be excellent.

**Open questions:**  
Research into "training-free multi-shot" approaches (sharing features across shots without per-character fine-tuning) is active and promising. Whether the next 12–24 months produce consumer-ready consistency tools with comparable quality to fine-tuned approaches is a genuinely open question.

**Sources:**  
- Skywork AI, *Character Consistency in Generative AI*, September 2025, https://skywork.ai/blog/character-consistency-generative-ai/  
- NeurIPS 2024, RefDrop paper, https://proceedings.neurips.cc/paper_files/paper/2024/file/3b057de5a2e38bd8fa10201866c20dbf-Paper-Conference.pdf  
- Medium (SophieZ), *How I Solved Character Consistency in ComfyUI*, March 2026, https://medium.com/@sophie_62065/how-i-solved-character-consistency-in-comfyui

---

### What LLMs Can and Cannot Do in a Design Context

**Facts:**  
LLMs perform well in design-adjacent tasks with documented professional use: UX copy and microcopy generation; naming and tagline development; brief summarisation and synthesis; accessibility audit prompting (checking copy against WCAG guidelines by description); competitive research synthesis; user research analysis and pattern identification; writing and editing design documentation; generating user story and acceptance criteria drafts. LLMs perform poorly on: spatial reasoning (they have no concept of physical layout); precise typographic decisions; colour relationships; anything requiring visual perception of an actual design; numerical layout specifications.

**Interpretation:**  
The pattern is consistent with LLM strengths more broadly: language-first tasks, synthesis, generation from constraints. Where designers underuse LLMs is in the language work that surrounds and supports design — the brief, the rationale, the client communication, the research synthesis. Where they over-apply LLMs is in asking them to do work that is inherently visual or spatial. A useful mental model: LLMs are the world's most capable research assistant and copywriter, available at all times. They are not a visual design tool.

**Open questions:**  
Multimodal models (which can perceive images) have begun to narrow the visual gap — Claude, GPT-4o, and Gemini can describe and respond to actual design work. Whether this extends to genuine layout reasoning and visual critique rather than surface-level description is actively developing.

**Sources:**  
- Figma 2025 AI Report data, https://www.figma.com/reports/ai-2025/  
- Autodesk AI Jobs in Design and Make 2025 Report, https://adsknews.autodesk.com/en/news/ai-jobs-report/

---

### The Current State of AI Video for Designers (April 2026)

**Facts:**  
The AI video landscape moved rapidly in 2025–26. Key milestones include: Sora (OpenAI, first previewed February 2024) was discontinued on 26 April 2026 — the same day this research was compiled — due to unsustainable compute costs (estimated $4.2M/day GPU costs against limited revenue). The market has consolidated around several surviving tools: **Veo 3.1** (Google DeepMind) is currently the most technically capable model, producing true 4K video with native audio-video synchronisation in a single pass; **Kling 3.0** (Kuaishou, China) offers 4K output with 2-minute clip capability, native text rendering, and the most cost-effective API pricing (~$0.50/clip); **Runway Gen-4.5** is valued for creative control and professional workflow integration; **Seedance 2.0** (ByteDance) features an "Identity Lock" system for consistent character appearance. As of early 2026, four of the six major models generate synchronised audio natively — a capability that did not exist in early 2025. Character consistency across shots has improved significantly but remains imperfect. Most professional users now operate across multiple tools, routing different shots to different models.

**Interpretation:**  
The trajectory has been faster than most practitioners expected. In 2024, AI video produced blurry 15-second clips with melting anatomy. By early 2026, the output quality rivals professionally produced B-roll for many use cases. For designers, this has immediate relevance in: presentation and pitch video; social media content; concept video for client communication; background and ambient material. The limitations that remain are primarily around sustained narrative coherence across long-form content and fine-grained art direction (the AI interprets, it doesn't follow exact instructions). Sora's closure is a significant data point: high capability without a sustainable business model is not enough. The market is not settled.

**Open questions:**  
Character consistency in video remains harder than in stills. Long-form narrative coherence (scenes that make visual sense shot-to-shot across a 2-minute piece) is improving but not solved. The regulatory environment under EU AI Act Article 52 (watermarking, provenance metadata for synthetic video, full effect August 2025) is adding compliance complexity.

**Sources:**  
- Wikipedia, *Sora (text-to-video model)*, accessed April 2026, https://en.wikipedia.org/wiki/Sora_(text-to-video_model)  
- Medium (Cliprise), *The State of AI Video Generation in February 2026*, https://medium.com/@cliprise/the-state-of-ai-video-generation-in-february-2026  
- Lushbinary, *AI Video Generation 2026: Sora 2 vs Veo 3.1 vs Kling 3.0*, https://lushbinary.com/blog/ai-video-generation-sora-veo-kling-seedance-comparison/  
- eWeek, *Sora Is Gone: Here Are 6 AI Video Tools Filling the Void in 2026*, https://www.eweek.com/news/sora-alternatives-ai-video-tools-2026/

---

### The Current State of AI for UI/UX — What the Tools Actually Do

**Facts:**  
Several distinct categories of tool now exist for AI-assisted UI/UX work:  
- **Figma AI** (integrated into Figma since 2024, significantly expanded at Config 2025): offers text-to-UI ("First Draft"), AI copy generation, layer renaming, visual search, background removal, Add Interactions, Dev Mode AI code generation, and (new 2025) "Figma Make" for working prototypes from prompts. Figma Sites (turning designs into live websites) launched in beta at Config 2025 and was immediately criticised for generating inaccessible code.  
- **v0 by Vercel**: Generates production-ready React + Tailwind UI components from text prompts or Figma designs/screenshots. Strong on frontend aesthetics; no backend generation.  
- **Lovable**: Full-stack app generation (UI + backend + database + auth + deployment) from natural language. Fastest MVP creation; limited customisation; vendor lock-in to React/Supabase stack.  
- **Bolt**: Browser-based AI app builder with real-time preview and backend integration. Fast prototyping; interface rougher than Lovable; supports more frameworks.

Figma's own 2025 AI report found that 85% of designers said AI skills would be essential to future success; only around half felt AI made them better at their jobs (versus more efficient). Figma Sites generated significant community criticism for accessibility failures at launch.

**Interpretation:**  
The most important distinction for a designer is between *prototype tools* (v0, Lovable, Bolt) and *design workflow tools* (Figma AI). Prototype tools are genuinely useful for rapid validation — they produce working code that can be tested with users — but they produce output that is heavily opinionated about tech stack (React + Tailwind + shadcn by default) and struggles with custom design systems. Design workflow tools (Figma AI) are embedded in existing practice and reduce friction on specific tasks, but their most hyped features (First Draft, Figma Make) produce starting points that require significant refinement, not finished output. Figma Sites's accessibility failures illustrate the risk of treating AI-generated production output as done.

**Open questions:**  
Whether "vibe design" (describe a UI in natural language, get a working design) becomes a mainstream designer workflow or remains a prototyping shortcut for non-designers is genuinely unclear. The tool quality is advancing rapidly. Whether it competes with or complements deep design expertise is the more interesting question.

**Sources:**  
- Figma, *Figma AI* feature page, accessed April 2026, https://www.figma.com/ai/  
- LogRocket, *Figma AI in 2026: Everything it can do — and what it still can't*, April 2026, https://blog.logrocket.com/ux-design/figma-ai-2026-quick-overview/  
- Forrester, *Figma Config 2025: In An AI World, Design Matters More Than Ever*, May 2025, https://www.forrester.com/blogs/figma-config-2025-in-an-ai-world-design-matters-more-than-ever/  
- NxCode, *Bolt vs Lovable vs V0: Which One to Choose in 2026?*, January 2026, https://uibakery.io/blog/bolt-vs-lovable-vs-v0  
- Till Freitag, *Lovable vs Bolt vs v0*, February 2026, https://till-freitag.com/en/blog/lovable-vs-bolt-vs-v0-en

---

## 3. TRAJECTORY — The Shape of the Curve

---

### The Timeline of Image Generation: What the Pace Actually Looks Like

**Facts:**  
A chronology of the key milestones:
- **January 2021**: DALL-E 1 (OpenAI) — proof of concept; outputs crude, clearly AI-generated
- **April 2022**: DALL-E 2 — significant quality jump; photorealistic outputs in some prompts; still widely impractical
- **February 2022**: Midjourney V1 enters limited beta; outputs "wonky" by current standards
- **August 2022**: Stable Diffusion 1.4 released open-source — marks democratisation of image generation; anyone can run locally
- **November 2022**: Midjourney V4 — first version widely considered "actually usable for creative work"
- **March 2023**: Midjourney V5 — photorealistic rendering; broader stylistic range
- **October 2023**: DALL-E 3 — significantly improved prompt adherence and text rendering
- **December 2023**: Midjourney V6 — first version to render text within images; major improvement in prompt interpretation
- **July 2024**: Midjourney V6.1 — ~25% speed increase; more coherent detail; still default until June 2025
- **April 2025**: Midjourney V7 — breakthrough improvements in face/hand anatomy, consistency, and creative direction following; introduced Draft Mode (10x speed, half cost) for rapid iteration
- **2024–2025**: Flux (Black Forest Labs, founded by core Stable Diffusion researchers) emerges as leading open-weight model; GPT-4o Image Generation (OpenAI, 2025) produces high-fidelity images with text rendering natively in ChatGPT

The interval between Midjourney V1 and V7 is 37 months. The visual quality difference is not incremental — it is the difference between outputs that read as obviously artificial and outputs that, in many cases, are indistinguishable from professional photography or illustration.

**Interpretation:**  
The timeline reveals something important: the pace has not been linear. The 2022–2023 period saw enormous leaps. The 2023–2025 period saw continued improvement but on specific problem areas (text, anatomy, consistency) rather than wholesale capability jumps. This is a pattern consistent with S-curve adoption: rapid initial gains as the core technology proves itself, then progressively more targeted improvements. It does not mean progress is slowing — it means the frontier is moving from "can it produce images" to "can it produce the *right* images reliably." Whether the curve will flatten meaningfully in the next three years, or whether new architectural approaches (transformer-based diffusion, reasoning-integrated generation) will produce another step change, is genuinely unknown.

**Open questions:**  
Whether model improvements will continue at this pace, and whether open-weight models will remain competitive with proprietary ones (the Stable Diffusion / Flux ecosystem versus Midjourney / DALL-E), are both genuinely uncertain. Legal risk (ongoing copyright litigation) could affect the training data available for future models.

**Sources:**  
- Midjourney official documentation (version history), https://docs.midjourney.com/hc/en-us/articles/32199405667853-Version  
- Aituts, *All Midjourney Versions (V1-V6) Compared*, https://aituts.com/midjourney-versions/  
- Midjourneyai.online, *Midjourney Versions V1–V7: A Complete Guide (2025)*, https://midjourneyai.online/midjourney-versions-complete-guide/  
- Britannica, *Midjourney*, https://www.britannica.com/technology/Midjourney

---

### When AI Entered Design Tools: Acceleration or Catch-Up?

**Facts:**  
Key integration dates across major design platforms:
- **March 2023**: Adobe launches Firefly beta (trained on licensed Stock content; IP-safe positioning)
- **Mid-2023**: Canva integrates generative AI features across its platform (Magic Media, Magic Write, Magic Design)
- **July 2024**: Figma AI launches as integrated feature set (free during beta throughout 2024); initial Make Designs feature temporarily pulled after producing outputs that resembled existing apps
- **Config 2025 (May 2025)**: Figma unveils four new AI-powered products: Figma Make, Figma Sites, Figma Buzz, Figma Draw
- **2025**: Figma's MCP server integration enables Figma design context to feed directly into AI coding environments (Claude, Cursor, VS Code, Windsurf)

Adobe's Content Credentials initiative — embedding tamper-evident provenance metadata in all Firefly outputs — positions Adobe as the "responsible AI" corporate actor in the creative tools market. The Content Authenticity Initiative (CAI) it co-founded has industry membership but adoption is uneven outside Adobe's own products.

**Interpretation:**  
This looks more like acceleration than catch-up. Adobe and Figma were integrating AI at roughly the same pace as enterprise software broadly — neither was a first mover at the model level, but both are now deeply embedding AI into professional design workflows. The distinction to note: these integrations are largely assistive (making existing tasks faster) rather than transformative (replacing professional creative judgment). Figma's strategic framing at Config 2025 — "design will be the differentiator as AI makes it easier to build software" — positions AI as raising the stakes for design quality, not eliminating the need for it. Whether this framing holds depends on how capable automated alternatives become.

**Open questions:**  
Figma Sites' accessibility failures are a data point worth watching: they suggest that AI-generated production output for even medium-complexity tasks still requires professional review. Whether this narrows significantly in the next two years will tell us a lot about whether "AI does it, designer reviews" becomes a viable professional workflow.

**Sources:**  
- Figma, *Meet Figma AI*, July 2024, https://www.figma.com/blog/introducing-figma-ai/  
- Forrester, *Figma Config 2025*, May 2025, https://www.forrester.com/blogs/figma-config-2025-in-an-ai-world-design-matters-more-than-ever/  
- Adobe Firefly product page, https://business.adobe.com/products/firefly-business/firefly-ai-approach.html

---

### Exponential Intuition: The Cognitive Trap in Reading AI Progress

**Facts:**  
A recurring problem in AI capability assessment is that humans are poorly calibrated for exponential change. The standard intuition treats capability on a linear scale: if a model is "30% as capable as a human designer" today, it feels like 70% remains to close — a large gap suggesting years of remaining distance. But on an exponential curve, closing the first 70% of capability may take two years, and closing the next 20% may take six months. The BLS's own employment projections methodology acknowledges this: they note that "job displacement due to technological change typically takes longer than technologists expect" — but also that AI may affect markets faster than historical technological transitions. A Stanford University study published August 2025 found a 13% decline in entry-level tech roles exposed to AI automation since 2022, with a measurable effect within three years of model deployment.

**Interpretation:**  
The most dangerous mental model for a designer is "AI is impressive but has obvious limitations, therefore I don't need to change my practice." The limitations are real. But the history of the last four years shows that specific, well-documented limitations (text rendering, hands, consistency, anatomy) have been partially or largely addressed within 12–24 months of being identified. Reading the current limitation list as a static catalogue of permanent incapabilities will produce systematic underestimation of where tools will be when a designer who is ignoring this now has to face it. The productive alternative is not alarm — it is maintaining active familiarity with the frontier so that shifts are legible when they happen.

**Open questions:**  
Whether the S-curve flattens meaningfully in the next three years is genuinely unknown. The rate of improvement on visual AI in particular has surprised most forecasters; there is no strong reason to expect it will stop surprising them.

**Sources:**  
- BLS, *Industry and Occupational Employment Projections 2024–34*, https://www.bls.gov/opub/mlr/2026/article/industry-and-occupational-employment-projections-overview.htm  
- Fortune, *BLS revisions: AI is automating away tech jobs*, September 2025, https://fortune.com/2025/09/09/bls-revisions-nearly-a-million-fewer-jobs-ai-automating-tech/

---

## 4. LANDSCAPE — Tools, Players, Current State

*Data reflects April 2026. This section will require the most frequent updating.*

---

### Midjourney (Current State: April 2026)

**Facts:**  
Midjourney V7 (released April 2025) became default in mid-2025 and remains current as of April 2026. Key features: breakthrough quality in face and hand rendering; improved character consistency across renders (previously a major weakness); Draft Mode (10x speed, half cost for rapid iteration); native web interface (launched August 2024 with V6.1, no longer Discord-only). Pricing: subscription tiers from $10–$120/month. V6.1 description (previous default until June 2025): 25% faster than V6, more coherent images with precise detail. Midjourney is an independent company with ~11 staff and has been profitable since 2022. It does not have open-source models. In August 2025, Meta announced a partnership to license Midjourney technology for image creation on Facebook and Instagram. Major studios (Disney/Universal in June 2025, Warner Bros. in September 2025) have filed copyright infringement lawsuits.

**Interpretation:**  
Midjourney remains the tool most associated with high-aesthetic AI image generation among creative practitioners — its default style is distinctive and has shaped a recognisable "AI visual aesthetic" across the industry. Its progression from wonky prototype (V1, 2022) to competitive creative tool (V7, 2025) across 37 months is the clearest benchmark for the pace of improvement in image AI. For designers: it is genuinely useful for mood-boarding, style exploration, and concept development; it is not reliable for brand-consistent asset production without significant technical investment; and its images carry recognisable aesthetic markers that make them identifiable to experienced eyes.

**Open questions:**  
The copyright litigation may affect training data availability for future versions. The Meta licensing deal may signal Midjourney's future direction — toward platform integration rather than standalone creative tool. Whether V8 will close the consistency gap further is unknown.

**Sources:**  
- Midjourney official documentation, https://docs.midjourney.com  
- Wikipedia, *Midjourney*, accessed April 2026, https://en.wikipedia.org/wiki/Midjourney  
- Britannica, *Midjourney*, https://www.britannica.com/technology/Midjourney

---

### Adobe Firefly and the IP-Safe Positioning

**Facts:**  
Adobe Firefly is trained exclusively on licensed Adobe Stock imagery, public domain content, and content Adobe owns — not on scraped web data. This is its principal market differentiator. Adobe offers IP indemnification to enterprise customers on qualifying plans for content generated with Firefly. Content Credentials are automatically attached to all Firefly outputs: tamper-evident metadata indicating the AI model used and version, embedded in the file. Adobe is a founding collaborator of the Content Authenticity Initiative (CAI), which promotes adoption of C2PA (Coalition for Content Provenance and Authenticity) standards across the industry. Enterprise Custom Models allow organisations to fine-tune Firefly on their own brand assets. Firefly is integrated across Creative Cloud (Photoshop, Illustrator, Premiere, Adobe Express) and available via API.

**Interpretation:**  
Adobe's strategy is coherent and intelligently positioned: it bets that enterprises and professionals will pay a premium for legal clarity, and that the creative market will increasingly require provenance metadata. The Content Credentials initiative positions Adobe as the infrastructure layer for a future where "was this made by AI?" is a relevant and traceable question. For designers working with brand clients, Firefly is currently the lowest-legal-risk option for AI-assisted production work. The trade-off is that its outputs are somewhat more conservative and less stylistically diverse than Midjourney or open-source alternatives, reflecting its more curated training data.

**Open questions:**  
Whether IP indemnification holds if a court eventually rules that training on licensed stock constitutes a more complex form of derivative work is unclear. The robustness of Adobe's legal positioning depends partly on legal developments that haven't yet been adjudicated.

**Sources:**  
- Adobe Firefly enterprise site, https://business.adobe.com/products/firefly-business/firefly-ai-approach.html  
- Adobe Content Credentials help, https://helpx.adobe.com/firefly/web/get-started/learn-the-basics/content-credentials-overview.html  
- Content Authenticity Initiative blog, https://contentauthenticity.org/blog/meeting-the-moment-with-c2pa-and-firefly

---

### v0, Lovable, and Bolt: What Designers Need to Know

**Facts:**  
These tools — along with Figma Make — constitute the emerging "vibe coding/vibe design" category: natural-language-to-functional-interface generation. The term "vibe coding" was coined by Andrej Karpathy in February 2025. Current state (April 2026):
- **v0 (Vercel)**: Generates production-ready React + Tailwind components from text prompts or uploaded Figma designs/screenshots. Best code quality in the frontend category; no backend. Pricing from $20/month; free tier available. Caused developer backlash in May 2025 after shifting from generous unlimited plan to metered model.
- **Lovable**: Full-stack (UI + backend + database + auth + deployment) from natural language. Reached $20M ARR in 2 months (fastest in European startup history). Best for complete MVP generation without code. Limited customisation; vendor lock-in to React + Supabase stack.
- **Bolt (StackBlitz)**: Browser-based full-stack coding environment with AI assistance. Fastest generation; supports more frameworks than Lovable; interface rougher.

All three tools default to React + Tailwind + shadcn/ui — a deliberate stack choice that enables their "intuition" about good UI defaults. Custom design systems work poorly unless explicitly injected via knowledge files or component libraries.

**Interpretation:**  
For designers, the key question is: does this threaten the design-to-dev handoff, or does it create new roles? The honest answer is: both, to different degrees. These tools can generate a functional prototype from a Figma file or a text description fast enough to change the conversation about whether a designer needs to be involved in early-stage product validation. The output quality is improving rapidly. But the tools produce "generic-but-functional" results that require significant design refinement to meet production standards. They are currently best understood as accelerating the *definition* stage of design (what are we building?) while leaving the *craft* stage (is this good?) firmly in human hands. As of April 2026, professional practitioners report a common workflow: v0 for components → Lovable for the overall app → Claude Code (or similar) for production cleanup.

**Open questions:**  
Whether these tools will eventually produce output that competes with senior design quality (not just functionality) depends on advances in both model capability and design knowledge encoding. The "vibe design" movement (describing UI in natural language, iterating on vibes) is reportedly where vibe coding was in early 2025 — a real movement with real tools, at an early stage.

**Sources:**  
- NxCode, *Bolt vs Lovable vs V0: Which One to Choose in 2026?*, January 2026, https://uibakery.io/blog/bolt-vs-lovable-vs-v0  
- Till Freitag, *Lovable vs. Bolt vs. v0*, February 2026, https://till-freitag.com/en/blog/lovable-vs-bolt-vs-v0-en  
- NxCode, *Vibe Design Tools 2026*, March 2026, https://www.nxcode.io/resources/news/vibe-design-tools-compared-stitch-v0-lovable-2026

---

## 5. PROFESSIONAL IMPLICATIONS — What This Means for Designers

---

### Employment Data for Designers: What the Numbers Actually Show

**Facts:**  
The U.S. Bureau of Labor Statistics (BLS) 2024–34 employment projections (published August 2025) explicitly identify graphic designers as an occupation expected to see *slower than average employment growth* due to AI productivity effects. The BLS states that generative AI "can also be used to automate repetitive tasks and speed up certain processes, such as prototyping and design verification" — effects expected to "limit demand" for graphic designers specifically. Arts, design, media, and communication occupations are classified as "particularly susceptible to productivity effects from generative AI." This is a productivity-suppressing-demand effect, not immediate displacement: the model assumes the number of things designed will not grow proportionally to productivity gains.

From BLS and adjacent research: the Stanford study (August 2025) found a 13% decline in entry-level tech roles exposed to AI automation since 2022. Autodesk's AI Jobs in Design and Make 2025 report found that AI skills mentions in job postings grew 114.8% in 2023, 120.6% in 2024, and 56.1% year-to-date in 2025 — with design overtaking technical expertise as the most in-demand skill in AI-related job postings. Entry-level tech roles in the UK fell 46% in 2024. BLS data revised downward by ~991,000 jobs for March 2024–March 2025, with the information sector (which includes tech) revised down by 88,000 (3%).

**Interpretation:**  
The picture is differentiated by level and task type. Entry-level roles — particularly those focused on asset production, templated design, and basic UI generation — are materially at risk on a 5–10 year horizon. Mid-to-senior design roles that involve strategic direction, systems thinking, stakeholder management, and creative judgment are significantly less exposed. The Autodesk data point (design overtaking technical skill in AI job postings) is interesting: it suggests that in an AI-native workflow, the bottleneck has shifted from "can you execute?" to "do you know what good looks like?" This is arguably a validation of the value of design judgment, rather than its erosion — but it also means designers who can't articulate that judgment or work within AI-assisted workflows will be at a disadvantage.

**Open questions:**  
The BLS methodology explicitly assumes "structural change in the economy due to technology will follow its historical pattern" — gradual, and slower than technologists predict. If AI productivity gains in design are faster or more sudden than past technological transitions, the projections will underestimate displacement. The evidence on pace is genuinely mixed.

**Sources:**  
- BLS, *Industry and Occupational Employment Projections 2024–34*, Monthly Labor Review 2026, https://www.bls.gov/opub/mlr/2026/article/industry-and-occupational-employment-projections-overview.htm  
- Upjohn Research, *AI Exposure and the Future of Work*, https://research.upjohn.org/cgi/viewcontent.cgi?article=1319&context=reports  
- Autodesk, *AI job growth in Design and Make: 2025 report*, https://adsknews.autodesk.com/en/news/ai-jobs-report/  
- Fortune, *BLS revisions: AI is automating away tech jobs*, September 2025, https://fortune.com/2025/09/09/bls-revisions-nearly-a-million-fewer-jobs-ai-automating-tech/  
- IntuitionLabs, *AI's Impact on Graduate Jobs: A 2025 Data Analysis*, November 2025, https://intuitionlabs.ai/articles/ai-impact-graduate-jobs-2025

---

### Which Design Tasks Are Most Exposed, and Which Are Least

**Facts:**  
High-exposure tasks (most vulnerable to AI automation or productivity reduction): stock illustration and icon generation; basic UI layout generation; mood board creation; template-based graphic design; image background removal and basic retouching; copy-and-paste asset production for campaigns; basic web page design from a brief. Lower-exposure tasks (more durable): design strategy and systems thinking; stakeholder facilitation and design research; creative direction and taste-making; brand identity development and management; accessibility and inclusive design expertise; user research synthesis and insight generation; managing and critiquing AI-generated output.

From BLS and Accenture (2023 framework applied to 2025 data): occupations with "high exposure, high complementarity" (where AI augments rather than replaces) are associated with employment growth. Occupations with "high exposure, low complementarity" face genuine replacement risk. The distinction tracks well onto the design task list above: tasks that require human judgment to evaluate quality, or human presence to build relationships, are complementarity-high. Tasks that are primarily execution of defined specifications are not.

**Interpretation:**  
The honest read is that junior designers doing asset production work are in the highest-risk category. Senior designers doing creative direction and strategic design are in the lowest-risk category. The career implication is not "design is safe" but rather "design expertise that cannot be replaced requires actively cultivating the parts of practice that AI cannot yet do, and demonstrating that value clearly." Designers who can evaluate and direct AI output are more valuable than those who can produce manual output — because the bottleneck has shifted from production to curation and judgment.

**Open questions:**  
Whether "creative direction" remains genuinely beyond AI capability is contested — the models are improving at aesthetic judgment tasks. Whether the gap between what AI can generate and what clients will accept as professionally adequate continues to narrow is the most important open question for mid-career designers.

**Sources:**  
- BLS projections data and methodology, August 2025  
- CBRE Investment Management, *Gen AI's Impact on U.S. Employment and Office Space*, February 2026, https://www.cbreim.com/insights/articles/gen-ai-impact-on-us-employment-and-office-space

---

### The Creative Ownership Question: What Copyright Law Currently Says

**Facts:**  
The U.S. Copyright Office has consistently held (as of 2024–2025) that AI-generated content without meaningful human authorship is not copyrightable. In the landmark Thaler v. Vidal case, the Federal Circuit ruled that non-human authors cannot hold copyright. For AI-assisted work where a human makes "sufficiently creative" selections and arrangements, copyright may apply to the human-created elements. The UK Copyright Office has reached similar conclusions. Who owns the output of AI generation tools varies by platform terms of service: Midjourney, DALL-E (OpenAI), and Firefly all grant users rights to generated content subject to platform terms, but with restrictions on using outputs to train competing models. The legal picture around AI-generated work used in commercial client deliverables is not fully settled.

**Interpretation:**  
The practical implication for designers doing client work: AI-generated outputs are commercially usable under most platform terms, but do not have the copyright protection that human-created original work does. This matters for clients who want to protect their brand assets from imitation. An AI-generated logo or illustration can be copied by a competitor without the copyright protection that a human-originated design would carry. For most commercial work, this is a low-stakes issue; for brand identity work intended to be protected, it is material.

**Open questions:**  
Court cases in the US (in which the Copyright Office's position is being challenged) and evolving EU positions may shift this picture. Whether training data itself constitutes copyright infringement remains the most contested legal question in the entire AI industry.

**Sources:**  
- Getty Images statement on UK ruling, November 2025, https://newsroom.gettyimages.com/en/getty-images/getty-images-issues-statement-on-ruling-in-stability-ai-uk-litigation  
- Ropes & Gray, *Getty Image Loses Copyright Claim*, January 2026, https://www.ropesgray.com/en/insights/viewpoints/102lvxe/getty-image-loses-copyright-infringement-claim-against-stability-ai-in-uks-first

---

## 6. CITIZEN BASELINE — The World Designers Are Operating In

---

### The Copyright and AI Training Question: Current Legal Status

**Facts:**  
The foundational question — whether training AI models on copyrighted images and text constitutes copyright infringement — remains legally unresolved in most jurisdictions.

The UK's highest-profile case: Getty Images v Stability AI, decided by the UK High Court on 4 November 2025. Result: Stability AI largely won. Key findings: (1) Getty abandoned its primary copyright infringement claim because training took place outside the UK — meaning the UK court had no jurisdiction over the act of training. (2) The court rejected Getty's secondary copyright claim that Stable Diffusion itself constituted an "infringing copy." The model does not store training data; it stores weights. (3) The court found limited trademark infringement where outputs displayed Getty watermarks, for early model versions only. The ruling did NOT determine whether training AI on copyrighted images is lawful — that question was procedurally avoided.

US litigation continues: class action suits by artists (Sarah Andersen, Kelly McKernan, Karla Ortiz) against Stability AI, Midjourney, and DeviantArt; Disney and Universal filed suit against Midjourney in June 2025 (described as "a bottomless pit of plagiarism"); Warner Bros. filed against Midjourney in September 2025. The US Copyright Office published its report on AI and copyright (Part 3, Generative AI Training) in 2024–2025, finding that training on publicly available data may constitute fair use but that the question is not settled.

**Interpretation:**  
For designers, this matters in two ways. First, the commercial AI tools they use were built on a legal framework that may eventually be adjudicated as infringement — producing retroactive liability for the companies, and potential uncertainty about tool availability. Second, their own work may be in AI training datasets without their knowledge or consent. The absence of a transparency obligation means there is currently no way to know. The practical risk for individual designers using tools like Firefly (trained on licensed data) is lower than for those using tools trained on scraped web data.

**Open questions:**  
The US and EU will likely produce significant legal clarity in 2026–2027. The UK, as evidenced by its March 2026 government report, is taking a "wait and see" approach, watching how the US and EU resolve the question before legislating.

**Sources:**  
- Latham & Watkins, *Getty Images v. Stability AI: English High Court Rejects Secondary Copyright Claim*, November 2025, https://www.lw.com/en/insights/getty-images-v-stability-ai-english-high-court-rejects-secondary-copyright-claim  
- Mayer Brown, *Getty Images v Stability AI: What the High Court's Decision Means*, November 2025, https://www.mayerbrown.com/en/insights/publications/2025/11/getty-images-v-stability-ai-what-the-high-courts-decision-means-for-rights-holders-and-ai-developers  
- Wikipedia, *Midjourney* (litigation section), accessed April 2026

---

### Artist Opt-Out Initiatives: Have I Been Trained?, Nightshade, and the Artist Response

**Facts:**  
"Have I Been Trained?" (haveibeentrained.com) is a tool developed by Spawning AI that allows artists and rights holders to check whether their work appears in LAION datasets (the primary training data for Stable Diffusion and related models) and submit opt-out requests. Spawning has partnerships with some AI companies to honour these requests. Nightshade (University of Chicago, 2024) is a tool that adds imperceptible perturbations to images that cause AI models to mislearn from them — a form of data poisoning intended as an adversarial defence for artists. Glaze (the same team) adds perturbations that cause AI models to misidentify an artist's style. In tests, Nightshade caused models trained on poisoned data to produce degraded outputs on targeted image categories. ArtStation (owned by Epic Games) introduced an "AI Art" exclusion option for portfolios following significant community pushback in 2022–2023.

**Interpretation:**  
These initiatives represent a genuine attempt by artists to reassert agency over their work in a system that currently provides little legal protection. Their practical effectiveness is limited: opt-out requests may or may not be honoured by companies using LAION data; Nightshade requires artists to actively apply it to every image they publish; and the perturbations may become detectable with more robust training pipelines. The more significant development may be policy: Adobe's Content Credentials initiative, if widely adopted, would give artists cryptographic tools to attach "do not train" preferences to their work in a technically verifiable way.

**Open questions:**  
Whether opt-out mechanisms will have legal force under future legislation is unresolved. Whether Nightshade-style adversarial defences scale to practical protection across large datasets is an open technical question.

**Sources:**  
- Adobe Content Credentials, *Generative AI training and usage preference*, https://helpx.adobe.com/creative-cloud/apps/adobe-content-authenticity/generative-ai-training-preferences.html  
- Content Authenticity Initiative, https://contentauthenticity.org

---

## 7. POLITICAL LAYER — UK Focus

*Entries dated carefully. This layer moves.*

---

### UK Government AI Policy: The Copyright and AI Report (March 2026)

**Facts:**  
On 18 March 2026, the UK government published its Report on Copyright and Artificial Intelligence and an accompanying economic impact assessment — both required under the Data (Use and Access) Act 2025 (DUAA), which itself passed in 2025 following significant creative industry lobbying. Key findings and commitments:

- The government's previously preferred approach (Option 3: a broad text and data mining exception for AI training, with a rights-reservation opt-out mechanism) is **no longer on the table**. It was dropped after 88% of Citizen Space consultation respondents rejected it, and following intense creative industry opposition.  
- No alternative has been formally adopted. The government does not currently have a preferred option.  
- A "focused exception targeted at specific types of use" was the most popular alternative in consultation responses.  
- The UK creative industries contributed £146 billion GVA to the UK economy in 2024 (6% of GDP), growing at 2.5x the economy's rate, supporting 7% of all UK jobs.  
- The UK AI sector contributed approximately £12 billion GVA in 2024 but could add £20–90 billion by 2030.  
- The government's immediate commitments: commissioning further research; establishing working groups on independent and smaller creative organisations; a consultation on "digital replicas" (likeness rights) in summer 2026; plans for AI-generated content labelling requirements.  
- Protection for computer-generated works under s.9(3) of the CDPA (a provision giving copyright to "computer-generated works") is proposed for removal, though with continued monitoring.

The House of Lords Communications and Digital Committee report (6 March 2026) concluded the UK creative industries face a "clear and present danger" from GenAI trained on copyrighted content, called the existing copyright framework an international "gold standard", and recommended a licensing-first approach while ruling out the opt-out model.

**Interpretation:**  
The UK's position in March 2026 is essentially: no decision yet, watching the US and EU. This is described explicitly in the report — the government is waiting for the international landscape to clarify before legislating. For designers and their clients, this means: existing UK copyright law remains as-is; enforcement against overseas-trained models remains difficult without transparency obligations; and licensing routes are the practical (not yet legally mandated) mechanism for resolving the training data question. The creative industries have effectively blocked the opt-out model; what they get in return is unclear. The tensions between the UK's ambition to lead in AI and its ambition to protect creative industries has produced a stalemate.

**Open questions:**  
How the US (through copyright litigation) and EU (through Digital Single Market Directive implementation and evolving case law) resolve their own versions of this question will likely determine what the UK does. A further consultation on digital replicas in summer 2026 may produce the first concrete legislative output.

**Sources:**  
- UK Government, *Copyright and AI Progress*, written statement to Parliament, 18 March 2026, https://questions-statements.parliament.uk/written-statements/detail/2026-03-18/hcws1416  
- Fieldfisher, *UK government maintains status quo on AI and copyright*, March 2026, https://www.fieldfisher.com/en/services/intellectual-property/intellectual-property-blog/uk-government-maintains-status-quo-on-ai-and-copyr  
- Lewis Silkin, *Opt-out cop-out? UK Government rethinks its position*, March 2026, https://www.lewissilkin.com/insights/2026/03/24/opt-out-cop-out-uk-government-rethinks-its-position-on-copyright-and-ai-102mnx9  
- House of Lords Communications and Digital Committee, *AI, copyright and the creative industries*, March 2026, https://publications.parliament.uk/pa/ld5901/ldselect/ldcomm/267/267.pdf  
- Two Birds, *Copyright & AI in the UK: The Debate Rolls On*, March 2026, https://www.twobirds.com/en/insights/2026/uk/copyright-,-a-,-aiin-the-uk-the-debate-rolls-on  
- Bratby Law, *UK Copyright and AI Training: Exception Dropped in 2026*, April 2026, https://bratby.law/copyright-and-ai-training-exception-2026/

---

### International Comparison: EU and US in Brief

**Facts:**  
**EU**: The EU Digital Single Market Directive (DSM) introduced a text and data mining (TDM) exception that allows AI training on publicly available data, with an opt-out mechanism for rights holders who do not want their work used. The EU AI Act (applicable as of August 2025 for high-risk provisions) requires general-purpose AI models to publish summaries of training data used, enabling copyright enforcement. The EU is therefore more advanced than the UK in transparency requirements but has not resolved the underlying licensing question.

**US**: Copyright litigation (multiple class actions, the Disney/Universal v. Midjourney suit) is working its way through courts. The US Copyright Office has found that AI-generated content without meaningful human authorship is not copyrightable. No legislation specifically addressing AI training data has passed as of April 2026. The fair use doctrine remains the primary legal framework being tested in court.

**Interpretation:**  
The UK sits between two more decisive jurisdictions. The EU has legislation (if imperfectly enforced); the US has active litigation producing case law. The UK is watching both and has so far produced only process. For designers working with international clients or using tools built and hosted outside the UK, the most practically relevant regulatory developments are in the US (where most AI companies are incorporated) and the EU (where transparency requirements are most advanced).

**Open questions:**  
Whether the US courts produce a clear fair use ruling on AI training — which would effectively establish global precedent given the US location of the major AI companies — is the single most important legal development to watch.

**Sources:**  
- VWV, *Copyright and artificial intelligence: analysing the UK government's March 2026 reports*, March 2026, https://www.vwv.co.uk/insights/articles/copyright-and-artificial-intelligence-analysing-the-uk-governments-march-2026-reports/  
- Kluwer Copyright Blog, *Copyright and AI policy in the UK in 2025*, December 2025, https://legalblogs.wolterskluwer.com/copyright-blog/copyright-and-ai-policy-in-the-uk-in-2025/

---

*End of research brief. Total entries: 20 across 7 research areas. Research date: 26 April 2026.*
