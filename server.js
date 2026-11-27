
require("dotenv").config();

const express = require("express");
const fs = require("fs");
const path = require("path");

const app = express();

const PORT = process.env.PORT || 3000;
const OLLAMA_URL = "http://127.0.0.1:11434/api/chat";
const MODEL = "qwen2.5:3b";

const KNOWLEDGE_PATH = path.join(__dirname, "knowledge.json");

app.use(express.json({ limit: "1mb" }));
app.use(express.static(__dirname));

/* =========================================
   1. ЗАГРУЗКА БАЗЫ ЗНАНИЙ
========================================= */

let knowledge = {
    authors: []
};

function loadKnowledge() {
    try {
        if (!fs.existsSync(KNOWLEDGE_PATH)) {
            throw new Error("Файл knowledge.json не найден");
        }

        const raw = fs.readFileSync(KNOWLEDGE_PATH, "utf8");
        const parsed = JSON.parse(raw);

        if (!parsed || !Array.isArray(parsed.authors)) {
            throw new Error("В knowledge.json отсутствует массив authors");
        }

        knowledge = parsed;

        const authorsCount = knowledge.authors.length;

        const worksCount = knowledge.authors.reduce((total, author) => {
            return total + (Array.isArray(author.works) ? author.works.length : 0);
        }, 0);

        console.log("База знаний загружена.");
        console.log("Авторов:", authorsCount);
        console.log("Произведений:", worksCount);

        return true;

    } catch (error) {
        console.error("Ошибка загрузки базы знаний:", error.message);
        return false;
    }
}

loadKnowledge();

/* =========================================
   2. ПОДГОТОВКА ТЕКСТА ДЛЯ ПОИСКА
========================================= */

function normalize(text) {
    return String(text || "")
        .toLowerCase()
        .replace(/ё/g, "е")
        .replace(/[^а-яa-z0-9\s]/gi, " ")
        .replace(/\s+/g, " ")
        .trim();
}

function getWords(text) {
    return normalize(text)
        .split(" ")
        .filter(word => word.length > 1);
}

function levenshtein(a, b) {
    if (a === b) return 0;
    if (!a.length) return b.length;
    if (!b.length) return a.length;

    const matrix = [];

    for (let i = 0; i <= b.length; i++) {
        matrix[i] = [i];
    }

    for (let j = 0; j <= a.length; j++) {
        matrix[0][j] = j;
    }

    for (let i = 1; i <= b.length; i++) {
        for (let j = 1; j <= a.length; j++) {
            const cost = a[j - 1] === b[i - 1] ? 0 : 1;

            matrix[i][j] = Math.min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + cost
            );
        }
    }

    return matrix[b.length][a.length];
}

function similarity(a, b) {
    a = normalize(a);
    b = normalize(b);

    if (!a || !b) return 0;
    if (a === b) return 1;

    if (a.includes(b) || b.includes(a)) {
        return 0.95;
    }

    const distance = levenshtein(a, b);
    return 1 - distance / Math.max(a.length, b.length);
}

function tokenSimilarity(query, target) {
    const queryWords = getWords(query);
    const targetWords = getWords(target);

    if (!queryWords.length || !targetWords.length) {
        return 0;
    }

    let matched = 0;

    for (const word of queryWords) {
        let best = 0;

        for (const targetWord of targetWords) {
            best = Math.max(best, similarity(word, targetWord));
        }

        if (best >= 0.72) {
            matched++;
        }
    }

    return matched / queryWords.length;
}

/* =========================================
   3. ПОИСК АВТОРОВ И ПРОИЗВЕДЕНИЙ
========================================= */

function getAuthorText(author) {
    return [
        author.name,
        author.years,
        author.country,
        author.bio,
        ...(Array.isArray(author.facts) ? author.facts : [])
    ].join(" ");
}

function getWorkText(work) {
    const characters = Array.isArray(work.characters)
        ? work.characters.map(character => {
            if (typeof character === "string") return character;

            return [
                character.name,
                character.description
            ].join(" ");
        }).join(" ")
        : "";

    return [
        work.title,
        work.genre,
        work.years,
        work.summary,
        work.description,
        work.theme,
        work.idea,
        work.mainIdea,
        work.problems,
        characters,
        ...(Array.isArray(work.aliases) ? work.aliases : [])
    ].flat().join(" ");
}

function findBestWork(message) {
    const query = normalize(message);
    const results = [];

    for (const author of knowledge.authors) {
        const works = Array.isArray(author.works) ? author.works : [];

        for (const work of works) {
            const title = normalize(work.title);
            const aliases = Array.isArray(work.aliases)
                ? work.aliases
                : [];

            let score = similarity(query, title) * 0.8;

            score = Math.max(
                score,
                tokenSimilarity(query, title) * 0.85
            );

            for (const alias of aliases) {
                score = Math.max(
                    score,
                    similarity(query, alias) * 0.85,
                    tokenSimilarity(query, alias) * 0.85
                );
            }

            const authorName = normalize(author.name);

            if (query.includes(authorName) && authorName.length > 3) {
                score += 0.12;
            }

            const fullText = normalize(getWorkText(work));

            for (const word of getWords(query)) {
                if (word.length >= 4 && fullText.includes(word)) {
                    score += 0.015;
                }
            }

            results.push({
                author,
                work,
                score: Math.min(score, 1)
            });
        }
    }

    results.sort((a, b) => b.score - a.score);

    if (!results.length) return null;

    const best = results[0];
    const second = results[1];

    // Не выбираем произведение, если два варианта
    // подходят почти одинаково.
    if (
        best.score >= 0.45 &&
        second &&
        best.score - second.score < 0.035 &&
        normalize(best.work.title) !== normalize(second.work.title)
    ) {
        return {
            ambiguous: true,
            candidates: results
                .filter(item => best.score - item.score < 0.035)
                .slice(0, 3)
        };
    }

    if (best.score < 0.35) {
        return null;
    }

    return best;
}

function findBestAuthor(message) {
    const query = normalize(message);
    let best = null;

    for (const author of knowledge.authors) {
        const score = Math.max(
            similarity(query, author.name),
            tokenSimilarity(query, author.name)
        );

        if (!best || score > best.score) {
            best = { author, score };
        }
    }

    return best && best.score >= 0.65 ? best.author : null;
}

/* =========================================
   4. ПОДГОТОВКА ИНФОРМАЦИИ О ПРОИЗВЕДЕНИИ
========================================= */

function formatWork(author, work) {
    const characters = Array.isArray(work.characters)
        ? work.characters.map(character => {
            if (typeof character === "string") {
                return character;
            }

            return `${character.name || ""}: ${character.description || ""}`;
        }).join("\n")
        : "";

    return `
АВТОР: ${author.name || "Не указан"}
ГОДЫ ЖИЗНИ: ${author.years || "Не указаны"}

ПРОИЗВЕДЕНИЕ: ${work.title || "Без названия"}
ЖАНР: ${work.genre || "Не указан"}
ГОД СОЗДАНИЯ: ${work.years || "Не указан"}

КРАТКОЕ СОДЕРЖАНИЕ:
${work.summary || work.description || "В базе нет готового пересказа."}

ГЕРОИ:
${characters || "Информация о героях отсутствует."}

ТЕМА:
${work.theme || "Не указана."}

ГЛАВНАЯ МЫСЛЬ:
${work.idea || work.mainIdea || "Не указана."}

ПРОБЛЕМАТИКА:
${Array.isArray(work.problems)
    ? work.problems.join(", ")
    : work.problems || "Не указана."}
`;
}

/* =========================================
   5. ПОИСК КРАТКОЙ СПРАВКИ В ИНТЕРНЕТЕ
========================================= */

async function wikipediaSearch(message) {
    try {
        const searchUrl = new URL(
            "https://ru.wikipedia.org/w/api.php"
        );

        searchUrl.searchParams.set("action", "query");
        searchUrl.searchParams.set("list", "search");
        searchUrl.searchParams.set("srsearch", message);
        searchUrl.searchParams.set("format", "json");
        searchUrl.searchParams.set("utf8", "1");
        searchUrl.searchParams.set("srlimit", "3");

        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 7000);

        const response = await fetch(searchUrl, {
            signal: controller.signal,
            headers: {
                "User-Agent": "KnowLendLiteratureAssistant/1.0"
            }
        });

        clearTimeout(timeout);

        if (!response.ok) return null;

        const data = await response.json();
        const results = data.query?.search || [];

        if (!results.length) return null;

        const best = results[0];

        const summaryUrl =
            "https://ru.wikipedia.org/api/rest_v1/page/summary/" +
            encodeURIComponent(best.title.replace(/ /g, "_"));

        const summaryController = new AbortController();
        const summaryTimeout = setTimeout(
            () => summaryController.abort(),
            7000
        );

        const summaryResponse = await fetch(summaryUrl, {
            signal: summaryController.signal,
            headers: {
                "User-Agent": "KnowLendLiteratureAssistant/1.0"
            }
        });

        clearTimeout(summaryTimeout);

        if (!summaryResponse.ok) return null;

        const summaryData = await summaryResponse.json();

        if (!summaryData.extract) return null;

        return {
            title: summaryData.title || best.title,
            extract: summaryData.extract,
            url: summaryData.content_urls?.desktop?.page ||
                `https://ru.wikipedia.org/wiki/${encodeURIComponent(best.title.replace(/ /g, "_"))}`
        };

    } catch (error) {
        console.log("Интернет-поиск недоступен:", error.message);
        return null;
    }
}

function looksLikeLiteratureQuestion(message) {
    const text = normalize(message);

    const keywords = [
        "книга",
        "произведен",
        "роман",
        "повест",
        "рассказ",
        "поэм",
        "стих",
        "автор",
        "писател",
        "поэт",
        "перескаж",
        "сюжет",
        "геро",
        "главная мысль",
        "смысл произведения",
        "биограф",
        "литератур"
    ];

    return keywords.some(word => text.includes(word));
}

/* =========================================
   6. ИСТОРИЯ ДИАЛОГА
========================================= */

function cleanHistory(history) {
    if (!Array.isArray(history)) return [];

    return history
        .filter(item =>
            item &&
            ["user", "assistant"].includes(item.role) &&
            typeof item.content === "string"
        )
        .slice(-10)
        .map(item => ({
            role: item.role,
            content: item.content.slice(0, 3000)
        }));
}

/* =========================================
   7. ОБРАЩЕНИЕ К ЛОКАЛЬНОЙ МОДЕЛИ
========================================= */

async function askOllama(message, history, context) {
    const systemPrompt = `
Ты — Литера, образовательный помощник сайта KnowLend.

ТВОИ ПРАВИЛА:

1. Всегда отвечай на русском языке.
2. Будь дружелюбной, понятной и внимательной.
3. Отвечай по существу, без лишних вступлений.
4. Понимай опечатки, ошибки и неточные названия.
5. Если пользователь просит пересказ, рассказывай последовательный сюжет произведения, а не общие рассуждения.
6. Не придумывай события, героев, даты и факты.
7. Если информация отсутствует в предоставленных материалах, честно сообщи об этом.
8. Не выдавай предположение за проверенный факт.
9. Если вопрос неоднозначный, задай уточняющий вопрос.
10. Используй обычное понятное оформление: абзацы, списки и небольшие заголовки.

ПРАВИЛА РАБОТЫ С БАЗОЙ:

Сначала используй информацию из раздела «ДАННЫЕ САЙТА».
Она является основным источником информации о произведениях каталога.

Если в базе есть краткое содержание, используй его для пересказа.
Если в базе нет содержания, не придумывай подробный сюжет.
Если есть только имя автора и название произведения, не делай вид, что знаешь весь текст.

ПРАВИЛА РАБОТЫ С ИНТЕРНЕТОМ:

Если предоставлена интернет-справка, используй только сведения из неё.
Не добавляй неподтверждённые детали.
Если справка короткая, ответ тоже должен быть коротким.

ДАННЫЕ САЙТА:
${context.local || "Подходящая запись в базе сайта не найдена."}

ИНТЕРНЕТ-СПРАВКА:
${context.internet || "Интернет-справка не найдена."}
`;

    const messages = [
        {
            role: "system",
            content: systemPrompt
        },
        ...history,
        {
            role: "user",
            content: message
        }
    ];

    const response = await fetch(OLLAMA_URL, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            model: MODEL,
            messages,
            stream: false,
            options: {
                temperature: 0.2,
                top_p: 0.8,
                repeat_penalty: 1.1,
                num_ctx: 4096
            }
        })
    });

    if (!response.ok) {
        throw new Error(`Ошибка Ollama: ${response.status}`);
    }

    const data = await response.json();

    const reply = data.message?.content;

    if (!reply) {
        throw new Error("Модель вернула пустой ответ");
    }

    return reply.trim();
}

/* =========================================
   8. API: КАТАЛОГ САЙТА
========================================= */

app.get("/api/catalog", (req, res) => {
    res.json({
        authors: knowledge.authors || []
    });
});

/* =========================================
   9. API: СОСТОЯНИЕ СЕРВЕРА
========================================= */

app.get("/api/status", (req, res) => {
    const authorsCount = knowledge.authors.length;

    const worksCount = knowledge.authors.reduce((total, author) => {
        return total + (
            Array.isArray(author.works) ? author.works.length : 0
        );
    }, 0);

    res.json({
        status: "ok",
        message: "Сервер KnowLend работает!",
        model: MODEL,
        authors: authorsCount,
        works: worksCount,
        knowledgeLoaded: authorsCount > 0
    });
});

/* =========================================
   10. API: ОБНОВЛЕНИЕ БАЗЫ ЗНАНИЙ
========================================= */

app.post("/api/reload-knowledge", (req, res) => {
    const success = loadKnowledge();

    if (!success) {
        return res.status(500).json({
            error: "Не удалось обновить базу знаний."
        });
    }

    res.json({
        success: true,
        message: "База знаний обновлена."
    });
});

/* =========================================
   11. API: ЧАТ С ЛИТЕРОЙ
========================================= */

app.post("/api/chat", async (req, res) => {
    try {
        const message = String(req.body?.message || "").trim();

        if (!message) {
            return res.status(400).json({
                error: "Напиши сообщение."
            });
        }

        if (message.length > 4000) {
            return res.status(400).json({
                error: "Сообщение слишком длинное."
            });
        }

        const history = cleanHistory(req.body?.history);

        let localContext = "";
        let internetContext = "";
        let sources = [];
        let foundWork = null;
        let notice = "";

        // Сначала ищем произведение в локальной базе.
        const workResult = findBestWork(message);

        if (workResult?.ambiguous) {
            const titles = workResult.candidates
                .map(item => `«${item.work.title}» — ${item.author.name}`)
                .join("\n");

            return res.json({
                reply:
                    "Я нашла несколько подходящих произведений. " +
                    "Уточни, какое именно ты имеешь в виду:\n\n" +
                    titles,
                sources: [],
                mode: "clarification"
            });
        }

        if (workResult) {
            foundWork = {
                title: workResult.work.title,
                author: workResult.author.name
            };

            localContext = formatWork(
                workResult.author,
                workResult.work
            );

            sources.push({
                title: "База знаний KnowLend",
                type: "local"
            });
        }

        // Если произведение не найдено, проверяем автора.
        if (!foundWork) {
            const author = findBestAuthor(message);

            if (author) {
                localContext = `
АВТОР: ${author.name}
ГОДЫ ЖИЗНИ: ${author.years || "Не указаны"}
СТРАНА: ${author.country || "Не указана"}

БИОГРАФИЯ:
${author.bio || "Биография отсутствует."}

ФАКТЫ:
${Array.isArray(author.facts)
    ? author.facts.join("\n")
    : "Дополнительных фактов нет."}

ПРОИЗВЕДЕНИЯ:
${(author.works || [])
    .map(work => `${work.title} (${work.years || "год неизвестен"})`)
    .join("\n")}
`;

                sources.push({
                    title: "База знаний KnowLend",
                    type: "local"
                });
            }
        }

        // Интернет используем только для литературных вопросов,
        // на которые не удалось найти ответ в базе.
        if (!localContext && looksLikeLiteratureQuestion(message)) {
            const internetResult = await wikipediaSearch(message);

            if (internetResult) {
                internetContext = `
Название: ${internetResult.title}

Информация:
${internetResult.extract}
`;

                sources.push({
                    title: internetResult.title,
                    url: internetResult.url,
                    type: "web"
                });

                notice =
                    "В моей базе сайта не нашлось подробного материала. " +
                    "Я использовала краткую справку из интернета.";
            } else {
                notice =
                    "В моей базе нет подходящей информации, " +
                    "а найти дополнительную справку сейчас не удалось.";
            }
        }

        const reply = await askOllama(
            message,
            history,
            {
                local: localContext,
                internet: internetContext
            }
        );

        res.json({
            reply,
            sources,
            notice,
            mode: localContext
                ? "local"
                : internetContext
                    ? "web"
                    : "general",
            foundWork
        });

    } catch (error) {
        console.error("Ошибка чата:", error);

        res.status(503).json({
            error:
                "Не удалось получить ответ от Литеры. " +
                "Проверь, запущена ли Ollama."
        });
    }
});

/* =========================================
   12. ЗАПУСК СЕРВЕРА
========================================= */

app.listen(PORT, () => {
    console.log("");
    console.log("=================================");
    console.log("       KnowLend успешно запущен!");
    console.log("=================================");
    console.log(`Сайт: http://localhost:${PORT}`);
    console.log(`Модель ИИ: ${MODEL}`);
    console.log("=================================");
    console.log("");
});
