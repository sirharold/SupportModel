Papers y Investigaciones Recientes (2022-2025)
1. Sistemas RAG para Soporte Técnico
"Retrieval Augmented Generation-Based Incident Resolution Recommendation System for IT Support" (2024)

Autores: Paulina Toro Isaza, Michael Nidd, et al.

Publicación: arXiv:2409.13707

Hallazgos clave: Los sistemas RAG resuelven dos problemas críticos en soporte técnico: cobertura de dominio y limitaciones de tamaño de modelo. El sistema combina RAG para generación de respuestas con modelos de clasificación y generación de consultas.

"Retrieval-Augmented Generation with Knowledge Graphs for Customer Service Question Answering" (2024)

Autores: Zhentao Xu, Mark Jerome Cruz, et al.

Publicación: arXiv:2404.17723

Resultados: Mejora del 77.6% en MRR y 0.32 en BLEU. Reducción del 28.6% en tiempo de resolución de incidentes en LinkedIn.

Metodología: Combina RAG con grafos de conocimiento para preservar estructura intra-issue e inter-issue.

2. Modelos de Transformers en Automatización de Soporte
"A Transformer-Based Approach for Smart Invocation of Automatic Code Completion" (2024)

Autores: A.D. de Moor, Arie van Deursen, M. Izadi

Contribución: Modelo de machine learning que predice cuándo invocar herramientas de completion automático basado en contexto de código y datos de telemetría.

Resultados: Superó significativamente el baseline manteniendo latencia baja.

3. Sistemas de Gestión de Conocimiento con NLP
"Using AI and NLP for Tacit Knowledge Conversion in Knowledge Management Systems" (2024)

Publicación: Preprints.org

Fecha: Diciembre 2024

Enfoque: Análisis comparativo de algoritmos NLP para minería de documentos y conversión de conocimiento tácito en sistemas de gestión de conocimiento.

"Natural Language Processing for IT Documentation and Knowledge Management" (2023)

Autor: Gireesh Kambala

Publicación: IJSRM Volume 11 Issue 02

Metodología: Aplicación de NLP para automatizar generación de documentación, mejorar recuperación de información y organizar repositorios de conocimiento.

4. Automatización de Help Desk y Service Desk
"Automation of Service Desk: Knowledge Management Perspective" (2021)

Autores: Michal Dostál, Jan Skrbek

Universidad: Technical University of Liberec

Propuesta: Modelo teórico de sistema automatizado de Service Desk que emplea técnicas de minería de texto, agente virtual, sistema experto y detector de intención del cliente.

"Natural Language Processing for Automated IT Service Desk Resolution" (2025)

Publicación: SSRN

Enfoque: Uso de NLP para automatizar respuestas de consultas de usuarios, categorización y resolución de problemas con mínima intervención humana.

5. Investigación en Generación de Respuestas Automáticas
"An Empirical Study of Corpus-Based Response Automation Methods for an E-mail-Based Help-Desk Domain" (2009)

Autores: Yuval Marom, Ingrid Zukerman

Universidad: Monash University

Metodología: Investigación de métodos basados en corpus para automatización de respuestas de help desk por email. Considera dos dimensiones: técnica de recopilación de información (retrieval vs prediction) y granularidad (documento vs sentencia).

Resultados: Los métodos combinados pudieron automatizar la generación de respuestas para 72% de las consultas de email.

6. Frameworks de Evaluación y Benchmarks
"Knowledge-Augmented Methods for Natural Language Processing" (2023)

Publicación: ACM International Conference on Web Search and Data Mining

Contribución: Tutorial sobre integración de conocimiento en NLP, incluyendo grounding desde texto, representación de conocimiento y fusión para aplicaciones de soporte técnico.

"Artefact Retrieval: Overview of NLP Models with Knowledge Base Access" (2022)

Autores: Vilém Zouhar, Marius Mosbach, et al.

Enfoque: Descripción sistemática de tipología de artefactos recuperados de bases de conocimiento, mecanismos de recuperación y métodos de fusión en modelos NLP.

Conferencias y Venues Académicos Relevantes
Conferencias Principales
EMNLP 2025: Call for papers incluye "Information Extraction and Retrieval" y "NLP Applications"

ACL 2025: Tutorials sobre campos relacionados con CL/NLP

NAACL-HLT 2025: Enfoque en tecnologías de lenguaje humano

Journals Especializados
Computational Linguistics

Journal of Information Retrieval

Expert Systems with Applications

Knowledge-Based Systems

Tendencias y Direcciones Futuras
1. Integración RAG-Knowledge Graphs
Los estudios más recientes muestran que la combinación de RAG con grafos de conocimiento mejora significativamente la precisión en recuperación de información técnica.

2. Modelos Multimodales
Investigación emergente en procesamiento de documentación que incluye texto, imágenes y diagramas técnicos.

3. Sistemas de Evaluación Automática
Desarrollo de métricas específicas para evaluar la calidad de respuestas automatizadas en contextos técnicos.

4. Personalización y Adaptación
Sistemas que se adaptan al nivel de expertise del usuario y contexto específico del problema.

Recursos y Herramientas de Investigación
Datasets y Benchmarks
ACL Anthology: Repositorio principal de investigación en NLP

Papers with Code: Implementaciones de RAG y sistemas de recuperación

Semantic Scholar: Plataforma de búsqueda académica con análisis de impacto

Herramientas de Desarrollo
Hugging Face: Modelos pre-entrenados para NLP técnico

LangChain: Framework para desarrollo de aplicaciones RAG

ChromaDB: Base de datos vectorial para sistemas de recuperación

Conclusiones
La investigación actual en uso de documentación oficial en soporte técnico está experimentando una transformación significativa hacia enfoques basados en IA generativa y sistemas híbridos. Los estudios más recientes (2024-2025) demuestran que:

Los sistemas RAG ofrecen soluciones viables para limitaciones de cobertura de dominio y tamaño de modelo

La combinación de grafos de conocimiento con RAG mejora sustancialmente la precisión en recuperación

Los modelos transformer son efectivos para automatización inteligente de procesos de soporte

La evaluación automática y métricas específicas son cruciales para validar efectividad

La dirección futura apunta hacia sistemas más inteligentes y adaptativos que pueden manejar documentación técnica compleja manteniendo alta precisión y experiencia de usuario personalizada.


LISTA DE LINKS: 
arxiv.org favicon
1. arXiv.org
arxiv.org/abs/2409.13707
Retrieval Augmented Generation-Based Incident Resolution ... - arXiv
Clients wishing to implement generative AI in the domain of IT Support and AIOps face two critical issues: domain coverage and model size constraints due to model choice limitations. Clients might choose to not use larger proprietary models such as GPT-4 due to cost and privacy concerns and so are limited to smaller models with potentially less domain coverage that do not generalize to the client's domain. Retrieval augmented generation is a common solution that addresses both of these issues: a retrieval system first retrieves the necessary domain knowledge which a smaller generative model leverages as context for generation. We present a system developed for a client in the IT Support domain for support case solution recommendation that combines retrieval augmented generation (RAG) for answer generation with an encoder-only model for classification and a generative large language model for query generation. We cover architecture details, data collection and annotation, development journey and preliminary validations, expected final deployment process and evaluation plans, and finally lessons learned.
arxiv.org favicon
2. arXiv.org
arxiv.org/abs/2404.17723
Retrieval-Augmented Generation with Knowledge Graphs for ... - arXiv
In customer service technical support, swiftly and accurately retrieving relevant past issues is critical for efficiently resolving customer inquiries. The conventional retrieval methods in retrieval-augmented generation (RAG) for large language models (LLMs) treat a large corpus of past issue tracking tickets as plain text, ignoring the crucial intra-issue structure and inter-issue relations, which limits performance. We introduce a novel customer service question-answering method that amalgamates RAG with a knowledge graph (KG). Our method constructs a KG from historical issues for use in retrieval, retaining the intra-issue structure and inter-issue relations. During the question-answering phase, our method parses consumer queries and retrieves related sub-graphs from the KG to generate answers. This integration of a KG not only improves retrieval accuracy by preserving customer service structure information but also enhances answering quality by mitigating the effects of text segmentation. Empirical assessments on our benchmark datasets, utilizing key retrieval (MRR, Recall@K, NDCG@K) and text generation (BLEU, ROUGE, METEOR) metrics, reveal that our method outperforms the baseline by 77.6% in MRR and by 0.32 in BLEU. Our method has been deployed within LinkedIn's customer service team for approximately six months and has reduced the median per-issue resolution time by 28.6%.
huggingface.co favicon
3. Huggingface
huggingface.co/papers/2405.14…
Paper page - A Transformer-Based Approach for Smart Invocation of Automatic Code Completion
Join the discussion on this paper page
preprints.org favicon
4. Preprints.org
preprints.org/manuscript/202…
Using AI and NLP for Tacit Knowledge Conversion in Knowledge Management Systems: A Comparative Analysis
Tacit knowledge, often implicit and embedded within individuals and organizational practices, plays a crucial role in knowledge management. Converting this tacit knowledge into explicit forms is vital for organizational effectiveness. This paper conducts a comparative analysis of NLP algorithms used for document and report mining in knowledge management systems (KMS) to facilitate the conversion of tacit knowledge. The focus is on algorithms that help extract tacit knowledge from documents and reports, as this knowledge is typically represented in semi-structured and document-based forms in natural language. NLP strategies, including text mining, information extraction, sentiment analysis, clustering, classification, recommendation systems, and affective computing, are explored for their effectiveness in this context. The paper provides a comprehensive analysis of the suitability of various NLP algorithms for tacit knowledge conversion within KMS, offering valuable insights for advancing research in this area.
5. ijsrm
ijsrm.net/index.php/ijsr…
[PDF] Natural Language Processing for IT Documentation and Knowledge ...
Gireesh Kambala, IJSRM Volume 11 Issue 02 February 2023 [www.ijsrm.net] EC-2023-964 With the increasing adoption of Artificial Intelligence (AI) and automation in IT workflows, Natural Language Processing (NLP) has emerged as a powerful tool to enhance IT documentation and knowledge management. NLP allows computers to understand, process, and generate human language, making it an ideal solution for improving technical documentation retrieval, summarization, classification, and organization....
pdfs.semanticscholar.org favicon
6. semanticscholar
pdfs.semanticscholar.org/c53b/574ea081d…
Automation of Service Desk: Knowledge Management Perspective
Michal Dost´al a and Jan Skrbek Department of Informatics, Faculty of Economics, Technical University of Liberec, Czech Republic Keywords: Service Desk, Automation, Knowledge Management. Abstract: In the current times, the need for quality and effective service support technologies is high. We can achieve good IT service operations that support good knowledge management practices by employing automation techniques and methods to the Service Desk systems. In our position paper, we look at the...
papers.ssrn.com favicon
7. papers.ssrn.com
papers.ssrn.com/sol3/papers.cf…
NATURAL LANGUAGE PROCESSING FOR AUTOMATED IT SERVICE DESK RESOLUTION
By automating the process of answering user inquiries, Natural Language Processing (NLP) has become a game-changing technology for improving IT service desk ope
aclanthology.org favicon
8. aclanthology
aclanthology.org/J09-4010.pdf
An Empirical Study of Corpus-Based Response Automation Methods for an E-mail-Based Help-Desk Domain
Response Automation Methods for an E-mail-Based Help-Desk Domain Yuval Marom∗ Monash University Ingrid Zukerman∗∗ Monash University This article presents an investigation of corpus-based methods for the automation of help-desk e-mail responses. Specifically, we investigate this problem along two operational dimensions: (1) information-gathering technique, and (2) granularity of the information. We consider two information-gathering techniques (retrieval and prediction) applied to information...
semanticscholar.org favicon
9. Annual Meeting of the Association for Computational Linguistics
semanticscholar.org/paper/Knowledg…
[PDF] Knowledge-Augmented Methods for Natural Language Processing | Semantic Scholar
This tutorial introduces the key steps in integrating knowledge into NLP, including knowledge grounding from text, knowledge representation and fusing, and introduces recent state-of-the-art applications in fusing knowledge into language understanding, language generation and commonsense reasoning. Knowledge in natural language processing (NLP) has been a rising trend especially after the advent of large scale pre-trained models. NLP models with attention to knowledge can i) access unlimited amount of external information; ii) delegate the task of storing knowledge from its parameter space to knowledge sources; iii) obtain up-to-date information; iv) make prediction results more explainable via selected knowledge. In this tutorial, we will introduce the key steps in integrating knowledge into NLP, including knowledge grounding from text, knowledge representation and fusing. In addition, we will introduce recent state-of-the-art applications in fusing knowledge into language understanding, language generation and commonsense reasoning.
image
arxiv.org favicon
10. arXiv.org
arxiv.org/abs/2201.09651
Overview of NLP Models with Knowledge Base Access - arXiv
Many NLP models gain performance by having access to a knowledge base. A lot of research has been devoted to devising and improving the way the knowledge base is accessed and incorporated into the model, resulting in a number of mechanisms and pipelines. Despite the diversity of proposed mechanisms, there are patterns in the designs of such systems. In this paper, we systematically describe the typology of artefacts (items retrieved from a knowledge base), retrieval mechanisms and the way these artefacts are fused into the model. This further allows us to uncover combinations of design decisions that had not yet been tried. Most of the focus is given to language models, though we also show how question answering, fact-checking and knowledgable dialogue models fit into this system as well. Having an abstract model which can describe the architecture of specific models also helps with transferring these architectures between multiple NLP tasks.
image
2025.emnlp.org favicon
11. EMNLP 2025
2025.emnlp.org/calls/main_con…
Call for Main Conference Papers - EMNLP 2025
Call for Main Conference Papers (EMNLP 2025)
2025.aclweb.org favicon
12. ACL 2025
2025.aclweb.org/calls/tutorials
Tutorials - ACL 2025 - Association for Computational Linguistics
ACL 2025 Call for Tutorials.
2025.emnlp.org favicon
13. EMNLP 2025
2025.emnlp.org/calls/papers/f…
First Call for Papers - EMNLP 2025
Official website for the 2025 Conference on Empirical Methods in Natural Language Processing
netapp.com favicon
14. netapp
netapp.com/media/115104-w…
Enhancing RAG Systems Lessons from Doc Development at NetApp
Enhancing RAG Systems Lessons from Doc Development at NetApp Grant Glass, NetApp October 2024 WP-7371 Abstract This white paper explores the development and refinement of a Retrieval-Augmented Generation (RAG) system named "Doc" at NetApp. It highlights the importance of a holistic approach to maximize answer relevancy and accuracy in RAG systems, focusing on three key areas: prompting strategies, retrieval mechanisms, and documentation improvements. The introduction emphasizes that the...
paperswithcode.com favicon
15. Paperswithcode
paperswithcode.com/task/retrieval…
Retrieval-augmented Generation | Papers With Code
749 papers with code • 0 benchmarks • 0 datasets These leaderboards are used to track progress in Retrieval-augmented Generation Large pre-trained language models have been shown to store factual knowledge in their parameters, and achieve state-of-the-art results when fine-tuned on downstream NLP tasks. Our framework trains a single arbitrary LM that adaptively retrieves passages on-demand, and generates and reflects on retrieved passages and its own generations using special tokens, called...
searchunify.com favicon
16. SearchUnify
searchunify.com/su/sudo-techni…
Revolutionizing the Customer Support with Transformer Models
Transformers are a type of deep learning architecture that empowers businesses to drive relevant and personalized customer support. How? Read on to know!
aclanthology.org favicon
17. aclanthology
aclanthology.org/2023.nlposs-1.…
Proceedings of the 3rd Workshop for Natural Language Processing Open Source Software (NLP-OSS 2023), pages 83–94
license.4 Development has been almost entirely volunteer-driven, though since 2021 the ACL has funded assistants who have contributed to inges- tions at the rate of about 20 hours a month. In this paper, we first describe the metadata cur- rently provided by the ACL Anthology and efforts to improve it (§2); the technical framework and de- velopment of the website (§3); as well as a Python library for accessing data from the Anthology (§4). We then look at the impact the ACL Anthology has had...
18. rebiun.org
rebiun.org/observatorio-d…
Herramientas de apoyo a la investigación
OsfOsf Open Science Framework (OSF) es una herramienta de gestión de proyectos gratuita y de código abierto que permite a los investigadores administrar, almacenar y compartir documentos, conjuntos de datos y otra información de forma transparente. Gratuita ConsensusConsensus Permite la búsqueda en diferentes bases de datos y repositorios en acceso abierto y ofrece resumen e información bibliométrica de los recurso. Versión gratuita limitada y de pago. Scopus AI Herramienta que genera breves...
intel.com favicon
19. intel
intel.com/content/dam/de…
Langchain Retrieval Augmented Generation White Paper
3 3. Explanation of RAG Pipeline 3.1. RAG Pipeline A typical RAG pipeline consists of several stages and may use more than one deep learning model in the question-answering process. An example full pipeline is shown below. Figure 2. The RAG pipeline consists of preparation stages that occur offline (before deployment) and active stages that occur when the user is interacting with the application. The first four stages occur offline, before the model is deployed: 1. The source documents are...
linkedin.com favicon
20. LinkedInEditors
linkedin.com/pulse/building…
Building a Local RAG Document Knowledge Base - LinkedIn
Disclaimer: The views expressed in this article are solely those of the author and do not necessarily reflect the views of any affiliated organizations. This content is for informational purposes only and should not be construed as professional advice.
image
Revisado
top2percentscientists.com favicon
Top 2% Scientists
top2percentscientists.com/best-nlp-resea…
Best NLP Research Papers of 2025
Discover the best NLP research papers of all time with clear visual aids, comprehensive summaries, and practical insights to power your 2025 projects.
Odin AI
blog.getodin.ai/using-ai-agent…
Using AI Agents For Technical Document Search: A Detailed Case ...
Transform your technical documentation with AI agents. Enhance document search, streamline processes, and boost efficiency.
thebestnlppapers.com favicon
The best NLP papers
thebestnlppapers.com
The best NLP papers of 2024 - The best NLP papers
Dive into the world of Natural Language Processing at thebestnlppapers.com. Our site ranks the best NLP papers each year based on citation count, providing a simple yet effective resource for researchers, students, and AI enthusiasts.
Journal of biomedical informatics
pmc.ncbi.nlm.nih.gov/articles/PMC27…
What can Natural Language Processing do for Clinical Decision ...
Computerized Clinical Decision Support (CDS) aims to aid decision making of health care providers and the public by providing easily accessible health-related information at the point and time it is needed. Natural Language Processing (NLP) is ...
cenia.cl favicon
Centro Nacional de Inteligencia Artificial
cenia.cl
CENIA
NOTICIAS Escucha el nuevo capítulo de Artificialmente Hablando: Sesgos en la Inteligencia Artificial: Desafíos y soluciones Junto a Gabriela Arriagada, profesora asistente del Instituto de Éticas Aplicadas UC, del IMC UC e investigadora joven de CENIA, y Vania Figueroa, Directora del la Unidad de Igualdad de Género de la Vicerrectoría de investigación y doctorados de […]
sciencedirect.com favicon
sciencedirect
sciencedirect.com/science/articl…
A decision support system for automating document retrieval and ...
jmir.org favicon
Journal of Medical Internet Research
jmir.org/2024/1/e55315
Clinical Decision Support and Natural Language Processing in ...
Background: Ensuring access to accurate and verified information is essential for effective patient treatment and diagnosis. Although health workers rely on the internet for clinical data, there is a need for a more streamlined approach. Objective: This systematic review aims to assess the current state of artificial intelligence (AI) and natural language processing (NLP) techniques in health care to identify their potential use in electronic health records and automated information searches. Methods: A search was conducted in the PubMed, Embase, ScienceDirect, Scopus, and Web of Science online databases for articles published between January 2000 and April 2023. The only inclusion criteria were (1) original research articles and studies on the application of AI-based medical clinical decision support using NLP techniques and (2) publications in English. A Critical Appraisal Skills Programme tool was used to assess the quality of the studies. Results: The search yielded 707 articles, from which 26 studies were included (24 original articles and 2 systematic reviews). Of the evaluated articles, 21 (81%) explained the use of NLP as a source of data collection, 18 (69%) used electronic health records as a data source, and a further 8 (31%) were based on clinical data. Only 5 (19%) of the articles showed the use of combined strategies for NLP to obtain clinical data. In total, 16 (62%) articles presented stand-alone data review algorithms. Other studies (n=9, 35%) showed that the clinical decision support system alternative was also a way of displaying the information obtained for immediate clinical use. Conclusions: The use of NLP engines can effectively improve clinical decision systems’ accuracy, while biphasic tools combining AI algorithms and human criteria may optimize clinical diagnosis and treatment flows. Trial Registration: PROSPERO CRD42022373386; https://www.crd.york.ac.uk/prospero/display_record.php?RecordID=373386
image
Stanfordnlp
nlp.stanford.edu
Stanford NLP Group - Stanford University
Performing groundbreaking Natural Language Processing research since 1999.
scispace.com favicon
scispace
scispace.com/pdf/document-r…
[PDF] Document Retrieval, Automatic - SciSpace
mdpi.com favicon
mdpi
mdpi.com/2076-3417/14/7…
Natural Language Processing in Knowledge-Based Support for ...
ibm.com favicon
ibm.com
ibm.com/es-es/think/to…
¿Qué es el PLN (procesamiento del lenguaje natural)? - IBM
El procesamiento del lenguaje natural (PLN) es un subcampo de la inteligencia artificial (IA) que utiliza el machine learning para ayudar a los ordenadores a comunicarse con el lenguaje humano.
surface.syr
surface.syr.edu/cgi/viewconten…
[PDF] Document Retrieval, Automatic - SURFACE at Syracuse University
ijcai.org favicon
ijcai
ijcai.org/Proceedings/81…
[PDF] A Knowledge-based Approach to Language Processing - IJCAI
onlinelibrary.wiley.com favicon
onlinelibrary.wiley
onlinelibrary.wiley.com/doi/abs/10.100…
The automatic retrieval of technical information - Wiley Online Library
sciencedirect.com favicon
sciencedirect
sciencedirect.com/journal/knowle…
New Avenues in Knowledge Bases for Natural Language Processing
oa.upm.es favicon
upm
oa.upm.es/75068/1/TFG_JA…
[PDF] Trabajo Fin de Grado La Inteligencia Artificial en la sociedad
2 Para ampliar esta información, el contenido del documento se ha estructurado de forma secuencial en cinco capítulos particulares, pero lógicamente relacionados, así: En primer lugar, se realiza una introducción general del documento que aborda el objeto y los objetivos de la investigación; este capítulo es la antesala al amplio recorrido investigativo que se realizó sobre el tema. En segundo lugar, se realizó un marco teórico-conceptual acerca de la definición, evolución histórica, tipos,...
diva-portal
diva-portal.org/smash/get/diva…
[PDF] Intelligent Document Retrieval System - DiVA portal
webs.uab.cat favicon
webs.uab.cat
webs.uab.cat/sibhilla/es/so…
Soporte técnico a la investigación
Realización de bases de datos bibliográficas y/o documentales y formación. Ofrecemos la posibilidad de realizar un estudio de les necesidades de información de un proyecto de investigación, diseño del sistema de gestión documental más idóneo para organizar y recuperar la información y formación de los técnicos y/o becarios que han de introducir los datos y/o gestionar el sistema. Diseño de aplicaciones web específicas para difundir los resultados de un proyecto o grupo de investigación. La...
editverse.com favicon
Editverse
editverse.com/natural-langua…
Natural Language Processing in Research: Analyzing Textual Data ...
Discover how Natural Language Processing in Research paves the way for groundbreaking insights from textual data in 2024. Explore the future now!
image
slideshare.net favicon
Slideshare
slideshare.net/slideshow/help…
Help desk system report
This document is a project report by Kumar Kartikeya Upadhyay on the development of an automated help desk system for a Bachelor of Technology degree. The system uses keyword search and case-based reasoning to improve information retrieval for users with varying expertise levels while ensuring easy maintenance through multiple classification ripple down rules (MCRDR). It discusses the role of expert systems, software development, and potential future enhancements for the project. - Download as a PDF or view online for free
codinsa.cl favicon
Codinsa
codinsa.cl/complete-guide…
Complete Guide to NLP in 2024: How It Works & Top Use Cases
8 NLP Examples: Natural Language Processing in Everyday Life These models were trained on large datasets crawled from the internet and web sources to automate tasks that require language understanding and technical sophistication. For instance, GPT-3 has been shown to produce lines of code based on human instructions. Natural language processing (NLP) is the technique […]
Pontificia Universidad Católica de Valparaíso
pucv.cl/pucv/noticias/…
Investigadora PUCV implementa plataforma con IA para mejorar la ...
La Inteligencia Artificial (IA) está cambiando el mundo de la educación. A través de diversas plataformas tecnológicas, los estudiantes y profesores pueden mejorar su redacción para tener un desempeño óptimo a la hora de redactar tesis u otros documentos de corte académico. En ese sentido, la académica del Instituto de Literatura y Ciencias del Lenguaje (ILCL) y estudiante del Doctorado en Lingüística de la PUCV, Constanza Cerda, participó en el desarrollo de la Plataforma para la Escritura...
aclanthology.org favicon
aclanthology
aclanthology.org/W06-0706.pdf
Proceedings of the Workshop on Task-Focused Summarization and Question Answering
Sydney, July 2006. c ⃝2006 Association for Computational Linguistics Automating Help-desk Responses: A Comparative Study of Information-gathering Approaches Yuval Marom and Ingrid Zukerman Faculty of Information Technology Monash University Clayton, VICTORIA 3800, AUSTRALIA {yuvalm,ingrid}@csse.monash.edu.au Abstract We present a comparative study of corpus- based methods for the automatic synthe- sis of email responses to help-desk re- quests. Our methods were developed by considering two...
investigauned.uned.es
investigauned.uned.es/herramientas-d…
Herramientas de IA para apoyar el proceso investigador
¿Estás utilizado ya alguna herramienta de IA para potenciar tu trabajo de investigación? Aquí te presentamos algunas que pueden ser especialmente útiles en las primeras fases del proceso investigador y ayudarte, tanto en la revisión bibliográfica y en la lectura de eficaz de textos académicos, como con la escritura de los manuscritos, generando textos o revisando los propuestos. ChatGPT, desarrollado por OpenAI, es un modelo de lenguaje avanzado diseñado para generar textos coherentes y...
recordskeeper.ai favicon
recordskeeper
recordskeeper.ai/features/smart…
Intelligent Document Search & Retrieval - RecordsKeeper.AI
unite.ai favicon
Unite.AI
unite.ai/es/leveraging-…
Aprovechar la IA generativa para la automatización de documentos
La automatización de documentos ha sido tradicionalmente competencia de los equipos legales y financieros, pero muchos más pueden beneficiarse de la creación de documentos automatizada mediante IA generativa. La atención al cliente, la investigación académica y otros sectores pueden disfrutar de las ventajas de la generación de documentos a gran escala, todo ello con la jerga específica del sector y adaptándose a diseños complejos que requieren una amplia gama de recursos.
aclanthology.org favicon
aclanthology
aclanthology.org/2024.acl-demos…
Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), pages 127–135
enhance the exploration experience by providing accurate responses to user queries (Zhu et al., 2024). Figure 1 illustrates how the knowledge graph and the LLM are integrated into our system. 3.1 Fields of Study Hierarchy Construction During exploration, users typically navigate from more well-known general concepts to less well- known and more specific concepts. Therefore, we use a semi-automated approach to construct a high-quality, hierarchical, acyclic graph of FoS in NLP. As a starting...
reply.com favicon
reply
reply.com/en/artificial-…
Maximising help desk efficiency with Generative AI - Reply
uclm.es favicon
Universidad de Castilla - La Mancha
uclm.es/areas/bibliote…
Inteligencia artificial para la investigación científica
Inteligencia artificial para la investigación científica
image
arxiv.org favicon
arxiv
arxiv.org/pdf/2106.15085…
Automatic Construction of Enterprise Knowledge Base
fresh without the need of any customized interven- tion. At the core of our knowledge base are entities mined from documents that are of interest to the en- terprise, such as product, organization and project. These entities are loosely referred to as topics to the end users (not to be confused with topic model- ing in NLP). The knowledge base is a collection of “topic cards” with rich information: properties that help users understand the topic (such as alternative names, descriptions, and...
teamhub.com favicon
teamhub
teamhub.com/blog/improving…
Improving Document Search and Retrieval Through Technology
arxiv.org favicon
arxiv
arxiv.org/pdf/2410.21306…
[PDF] Natural Language Processing for the Legal Domain - arXiv
3.1 Search Strategy We performed a systematic search across two academic databases to identify relevant studies, including: Google Scholar and IEEE Xplore. Then, search queries were crafted to capture studies that focused on the application of NLP to legal tasks. The search was defined by the following queries: • Query 1: (“Natural Language Processing” OR “NLP”) AND (“Legal” OR “Law”) • Query 2: (“Legal” AND (“Named Entity Recognition” OR “NER” OR “Document Summarization” OR “Text...
arxiv.org favicon
arXiv.org
arxiv.org/abs/2505.11933
Conversational Recommendation System using NLP and Sentiment ...
In today's digitally-driven world, the demand for personalized and context-aware recommendations has never been greater. Traditional recommender systems have made significant strides in this direction, but they often lack the ability to tap into the richness of conversational data. This paper represents a novel approach to recommendation systems by integrating conversational insights into the recommendation process. The Conversational Recommender System integrates cutting-edge technologies such as deep learning, leveraging machine learning algorithms like Apriori for Association Rule Mining, Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN), and Long Short-Term Memory (LTSM). Furthermore, sophisticated voice recognition technologies, including Hidden Markov Models (HMMs) and Dynamic Time Warping (DTW) algorithms, play a crucial role in accurate speech-to-text conversion, ensuring robust performance in diverse environments. The methodology incorporates a fusion of content-based and collaborative recommendation approaches, enhancing them with NLP techniques. This innovative integration ensures a more personalized and context-aware recommendation experience, particularly in marketing applications.
arxiv.org favicon
arxiv
arxiv.org/list/cs.CL/rec…
Computation and Language - arXiv
arXiv:2507.06085 [pdf, html, other] Title: A Survey on Prompt Tuning Zongqian Li, Yixuan Su, Nigel Collier Subjects: Computation and Language (cs.CL) arXiv:2507.06056 [pdf, html, other] Title: Entropy-Memorization Law: Evaluating Memorization Difficulty of Data in LLMs Yizhan Huang, Zhe Yang, Meifang Chen, Jianping Zhang, Michael R. Lyu Subjects: Computation and Language (cs.CL); Artificial Intelligence (cs.AI) arXiv:2507.06016 [pdf, html, other] Title: Conditional Multi-Stage Failure Recovery...
semanticscholar.org favicon
Data Analytics and Learning
semanticscholar.org/paper/Automate…
Automated IT Service Desk Systems Using Machine Learning Techniques | Semantic Scholar
Different classification algorithms like Multinomial Naive Bayes, Logistic regression, K-Nearest neighbor and Support vector machines are used to build such a ticket classifier system and performances of classification models are evaluated using various performance metrics. Managing problem tickets is a key issue in any IT service industry. The routing of a problem ticket to the proper maintenance team is very critical step in any service desk (Helpdesk) system environment. Incorrect routing of tickets results in reassignment of tickets, unnecessary resource utilization, user satisfaction deterioration, and have adverse financial implications for both customers and the service provider. To overcome this problem, this paper proposes a service desk ticket classifier system which automatically classifies the ticket using ticket description provided by user. By mining historical ticket descriptions and label, we have built a classifier model to classify the new tickets. A benefit of building such an automated service desk system includes improved productivity, end user experience and reduced resolution time. In this paper, different classification algorithms like Multinomial Naive Bayes, Logistic regression, K-Nearest neighbor and Support vector machines are used to build such a ticket classifier system and performances of classification models are evaluated using various performance metrics. A real-world IT infrastructure service desk ticket data is used for this research purpose. Key task in developing such a ticket classifier system is that the classification has to happen on the unstructured noisy data set. Out of the different models developed, classifier based on Support Vector Machines (SVM) performed well on all data samples.
Emnlp-2021
2021.emnlp.org/call-for-paper…
Style Files and Formatting | EMNLP 2021
The templates are available as an Overleaf template and can also be downloaded directly (LaTeX, Word). Please follow the formatting documentation general to *ACL conferences available here. The templates themselves contain only specific notes (e.g., LaTeX notes in the .tex file). Extra space for ethical considerations:Please note that extra space is allowed after the 8th page (4th page for short papers) for an ethics/broader impact statement. At submission time, this means that if you need...
arxiv.org favicon
arxiv
arxiv.org/pdf/2410.16498…
[PDF] arXiv:2410.16498v2 [cs.CL] 25 Mar 2025
Naoki Otani Nikita Bhutani Estevam Hruschka Megagon Labs {naoki,nikita,estevam}@megagon.ai Abstract The domain of human resources (HR) includes a broad spectrum of tasks related to natural lan- guage processing (NLP) techniques. Recent breakthroughs in NLP have generated signif- icant interest in its industrial applications in this domain and potentially alleviate challenges such as the difficulty of resource acquisition and the complexity of problems. At the same time, the HR domain can also...
citeseerx.ist.psu.edu favicon
psu
citeseerx.ist.psu.edu/document?repid…
This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.
IEEE TRANSACTIONS ON SYSTEMS, MAN, AND CYBERNETICS—PART B: CYBERNETICS 1 iHelp: An Intelligent Online Helpdesk System Dingding Wang, Tao Li, Shenghuo Zhu, and Yihong Gong Abstract—Due to the importance of high-quality customer service, many companies use intelligent helpdesk systems (e.g., case-based systems) to improve customer service quality. However, these systems face two challenges: 1) Case retrieval measures: most case-based systems use traditional keyword-matching-based ranking schemes...
aclrollingreview.org favicon
ACL Rolling Review
aclrollingreview.org/cfp
CALL FOR PAPERS – ACL Rolling Review
A peer review platform for the Association for Computational Linguistics
github.com favicon
GitHub
github.com/monologg/nlp-a…
nlp-arxiv-daily/ at master · monologg/nlp-arxiv-daily
Automatically Update NLP Papers Daily using Github Actions (ref: https://github.com/Vincentqyw/cv-arxiv-daily) - monologg/nlp-arxiv-daily
iaeng
iaeng.org/publication/WC…
Help Desk Management System
support for Information Technology Resource Center. In order to evaluate the scheme, it needs to characterize the performance in terms of quality of the output, time to process requests, and extent of usability. Specifically, it will seek to solve the following problems: 1. What support tools are available to the help desk and available to its clients? 2. How does the performance of proposed automated help desk measure the information? Assumptions Many of the daily tasks required by most...
github.com favicon
GitHub
github.com/acl-org/acl-st…
acl-org/acl-style-files: Official style files for papers submitted ... - GitHub
Official style files for papers submitted to venues of the Association for Computational Linguistics - acl-org/acl-style-files
papers.ssrn.com favicon
papers.ssrn.com
papers.ssrn.com/sol3/papers.cf…
NLLG Quarterly arXiv Report 09/24: What are the most influential current AI Papers?
The NLLG (Natural Language Learning & Generation) arXiv reports assist in navigating the rapidly evolving landscape of NLP and AI research across cs.CL, cs.
image
aaai.org favicon
aaai
aaai.org/papers/005-iaa…
Help Desk: Using AI to Improve Customer Servic - AAAI
github.com favicon
GitHub
github.com/UKPLab/arxiv-2…
arxiv-2024-nlp-contributions/README.md at main · UKPLab/arxiv-2024-nlp-contributions
This repository contains code and data for our paper. - UKPLab/arxiv-2024-nlp-contributions
cdn.aaai
cdn.aaai.org/ojs/11386/1138…
Hi, How Can I Help You?: Automating Enterprise IT Support Help Desks
papers.ssrn.com favicon
papers.ssrn.com
papers.ssrn.com/sol3/papers.cf…
Application of Transformer Models for Advanced Process Optimization and Process Mining
The exponential growth of data and increasing complexity of business processes necessitate advanced tools for process optimization and mining. Transformer model
papers.ssrn.com favicon
papers.ssrn.com
papers.ssrn.com/sol3/papers.cf…
Lightweight Retrieval-Augmented Generation System for Technical ...
This paper presents the development of an intelligent technical support system for enterprise resource planning platforms in the construction industry. Using a
image
puppyagent.com favicon
PuppyAgent
puppyagent.com/blog/RAG-Knowl…
Optimizing RAG Knowledge Bases for Enhanced Information Retrieval
Read our latest blog posts about AI knowledge base and RAG.
toolify.ai favicon
toolify.ai
toolify.ai/ai-news/unlock…
Unlocking the Power of Transformer Models in Automation and Creativity
Discover how transformer models revolutionize natural language processing and automate tasks, with insights from Ceyhun Derinbogaz at Unstructured @ Tilo.
paperswithcode.com favicon
Paperswithcode
paperswithcode.com/paper/retrieva…
Papers with Code - Retrieval Augmented Generation-Based Incident Resolution Recommendation System for IT Support
No code available yet.
learn.microsoft.com favicon
MicrosoftLearn
learn.microsoft.com/en-us/azure/se…
Retrieval Augmented Generation (RAG) in Azure AI Search
Learn how generative AI and retrieval augmented generation (RAG) patterns are used in Azure AI Search solutions.
image
ibm.com favicon
ibm.com
ibm.com/think/topics/t…
What is a Transformer Model? - IBM
A transformer model is a type of deep learning model that has quickly become fundamental in natural language processing (NLP) and other machine learning (ML) tasks.
image
arxiv.org favicon
arXiv.org
arxiv.org/abs/2402.19473
Retrieval-Augmented Generation for AI-Generated Content: A Survey
Advancements in model algorithms, the growth of foundational models, and access to high-quality datasets have propelled the evolution of Artificial Intelligence Generated Content (AIGC). Despite its notable successes, AIGC still faces hurdles such as updating knowledge, handling long-tail data, mitigating data leakage, and managing high training and inference costs. Retrieval-Augmented Generation (RAG) has recently emerged as a paradigm to address such challenges. In particular, RAG introduces the information retrieval process, which enhances the generation process by retrieving relevant objects from available data stores, leading to higher accuracy and better robustness. In this paper, we comprehensively review existing efforts that integrate RAG technique into AIGC scenarios. We first classify RAG foundations according to how the retriever augments the generator, distilling the fundamental abstractions of the augmentation methodologies for various retrievers and generators. This unified perspective encompasses all RAG scenarios, illuminating advancements and pivotal technologies that help with potential future progress. We also summarize additional enhancements methods for RAG, facilitating effective engineering and implementation of RAG systems. Then from another view, we survey on practical applications of RAG across different modalities and tasks, offering valuable references for researchers and practitioners. Furthermore, we introduce the benchmarks for RAG, discuss the limitations of current RAG systems, and suggest potential directions for future research. Github: https://github.com/PKU-DAIR/RAG-Survey.
docs.aws.amazon.com favicon
docs.aws.amazon
docs.aws.amazon.com/bedrock/latest…
Retrieve data and generate AI responses with Amazon Bedrock ...
repository.tudelft.nl
repository.tudelft.nl/record/uuid:f7…
A Transformer-Based Approach for Smart Invocation of Automatic Code Completion
A.D. de Moor (TU Delft - Software Engineering) Arie van Van Deursen (TU Delft - Software Engineering) M. Izadi (TU Delft - Software Engineering) ## More Info expand_more Other than for strictly personal use, it is not permitted to download, forward or distribute the text or part of it, without the consent of the author(s) and/or copyright holder(s), unless the work is under an open content license such as Creative Commons. Transformer-based language models are highly effective for code...
paperswithcode.com favicon
Paperswithcode
paperswithcode.com/paper/retrieva…
Papers with Code - Retrieval-Augmented Generation for AI-Generated Content: A Survey
Implemented in 3 code libraries.
docs.aws.amazon.com favicon
docs.aws.amazon
docs.aws.amazon.com/bedrock/latest…
How Amazon Bedrock knowledge bases work