

import os
import re
import uuid
import csv
import asyncio
import getpass
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

from sentence_transformers import SentenceTransformer, util
from langchain_google_genai import ChatGoogleGenerativeAI
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

# user utilities (assumed to be available)
from read_pdf import read_pdf
from summarize import summarize_text

# Global models
model = SentenceTransformer("all-MiniLM-L6-v2")
resume_embedding_store: Dict[str, Any] = {}
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

SIMILARITY_THRESHOLD = 0.5  # high-match threshold


def embed_resume_text(user_id: str, resume_text: str):
    print("🔎 Embedding resume ...")
    emb = model.encode(resume_text, convert_to_tensor=True)
    resume_embedding_store[user_id] = emb


def compare_job_with_resume_text(user_id: str, job_text: str) -> float:
    if user_id not in resume_embedding_store:
        return 0.0
    job_emb = model.encode(job_text, convert_to_tensor=True)
    sim = util.cos_sim(resume_embedding_store[user_id], job_emb).item()
    return float(sim)


def generate_cover_letter(resume_summary: str, job_summary: str) -> str:
    prompt = f"""
You are a professional assistant that writes concise, targeted cover letters (~180-220 words).
Resume summary:
{resume_summary}

Job summary:
{job_summary}

Write a professional cover letter emphasizing fit and relevant skills.
"""
    resp = llm.invoke(prompt)
    content = getattr(resp, "content", None) or str(resp)
    return content


async def click_and_extract_job_detail(page, card_element) -> str:
    """
    Clicks a job card element and extracts the job detail text.
    If clicking opens a new page, switch to it; else, scrape the current page.
    Returns textual representation of job detail.
    """
    try:
        await card_element.scroll_into_view_if_needed()
        await card_element.click(modifiers=["Control"], timeout=5000)
        context = page.context
        pages = context.pages
        if len(pages) > 1:
            detail_page = pages[-1]
            await detail_page.bring_to_front()
            await detail_page.wait_for_load_state("domcontentloaded", timeout=10000)
            text = await detail_page.locator("body").inner_text()
            return text, detail_page
        else:
            await page.wait_for_load_state("domcontentloaded", timeout=8000)
            text = await page.locator("body").inner_text()
            return text, page
    except Exception as e:
        try:
            href = await card_element.get_attribute("href")
            if href:
                if href.startswith("/"):
                    href = "https://www.ncs.gov.in" + href
                await page.context.new_page().goto(href)
                detail_page = page.context.pages[-1]
                await detail_page.wait_for_load_state("domcontentloaded", timeout=8000)
                text = await detail_page.locator("body").inner_text()
                return text, detail_page
        except Exception:
            pass
        try:
            return await card_element.inner_text(), page
        except Exception:
            return f"Error extracting job detail: {e}", page


def parse_job_text_for_fields(raw_text: str, job_url: str = "") -> Dict[str, Any]:
    """
    Try to parse title, company, posted on, last date, location, salary, skills, contact, description.
    Use regex heuristics, otherwise fall back to portions of raw_text.
    """
    text = re.sub(r'\r\n|\r', '\n', raw_text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()

    def extract(pattern):
        m = re.search(pattern, text, re.IGNORECASE)
        return (m.group(1).strip() if m else "")

    job_id = extract(r"Job Id\s*[:\-]?\s*([A-Za-z0-9\-\_]+)")
    title = extract(r"Job Title\s*[:\-]?\s*(.+?)\n")
    company = extract(r"Company\s*Name\s*[:\-]?\s*(.+?)\n")
    posted_on = extract(r"Posted On\s*[:\-]?\s*(.+?)\n")
    last_date = extract(r"Last date to apply\s*[:\-]?\s*(.+?)\n")
    salary = extract(r"Salary\s*[:\-]?\s*(.+?)\n")
    location = extract(r"Job Location\s*[:\-]?\s*(.+?)\n")
    skills = extract(r"Key Skills\s*[:\-]?\s*(.+?)\n")
    contact = extract(r"Contact Details\s*[:\-]?\s*(.+?)\n")

    desc_match = re.search(r"(Job Description[:\-\s]*\n(.+?))(?:\n[A-Z][a-z]+:|\Z)", text, re.IGNORECASE | re.DOTALL)
    if desc_match:
        description = desc_match.group(2).strip()
    else:
        description = text[:8000]

    return {
        "url": job_url,
        "job_id": job_id,
        "title": title,
        "company": company,
        "posted_on": posted_on,
        "last_date": last_date,
        "salary": salary,
        "location": location,
        "skills": skills,
        "contact": contact,
        "description": description,
        "raw_text_snippet": text[:400]
    }


async def run_agent_interactive(
    resume_text: str,
    login_with_creds: bool,
    ncs_username: str | None,
    ncs_password: str | None,
    sector: str,
    location: str,
    org_type: str,
    qualification: str,
    job_nature: str,
    no_jobs: int
):
    user_id = str(uuid.uuid4())
    embed_resume_text(user_id, resume_text)

    print("🌐 Launching browser (visible) ...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=500)
        context = await browser.new_context()
        page = await context.new_page()

        # 1) Login flow
        if login_with_creds:
            print("🔐 Navigating to NCS login page...")
            try:
                await page.goto("https://www.ncs.gov.in/_layouts/15/ncsp/user-management/login.aspx", wait_until="load", timeout=60000)
                
                # Save screenshot for debugging
                await page.screenshot(path="login_page.png")
                print("📸 Saved screenshot of login page to 'login_page.png'.")

                # Wait for form to load (less restrictive than role="textbox")
                try:
                    await page.wait_for_selector("input, [role='textbox']", state="visible", timeout=30000)
                    print("✅ Input or textbox elements are visible.")
                except PlaywrightTimeoutError:
                    print("⚠️ No input or textbox elements found within 30s. Possible CAPTCHA or redirect.")
                    await page.screenshot(path="login_no_inputs.png")
                    print("📸 Saved screenshot to 'login_no_inputs.png'.")

                # Fill username
                username_filled = False
                try:
                    username_locator = page.get_by_role("textbox", name="User Name")
                    await username_locator.wait_for(state="visible", timeout=10000)
                    await username_locator.click()  # Mimic sync code
                    await username_locator.fill(ncs_username)
                    print("👤 Filled username using get_by_role('textbox', name='User Name').")
                    username_filled = True
                except Exception as e:
                    print(f"⚠️ Failed to fill username with get_by_role('textbox', name='User Name'): {e}")
                    # Fallback selectors
                    fallbacks = [
                        (page.get_by_role("textbox", name=re.compile(r"User|Login|Email", re.I)), "accessible name regex"),
                        (page.locator("input[type='email'], input[type='text']"), "generic text/email input"),
                        (page.locator("input#ctl00_PlaceHolderMain_txtUserName"), "ASP.NET username ID"),
                        (page.locator("input[name='ctl00$PlaceHolderMain$txtUserName']"), "ASP.NET username name")
                    ]
                    for locator, desc in fallbacks:
                        try:
                            if await locator.count() > 0:
                                await locator.first.click()
                                await locator.first.fill(ncs_username)
                                print(f"👤 Filled username using fallback: {desc}")
                                username_filled = True
                                break
                        except Exception as fallback_e:
                            print(f"⚠️ Fallback {desc} failed: {fallback_e}")

                # Fill password
                try:
                    password_locator = page.get_by_role("textbox", name="Password")
                    await password_locator.wait_for(state="visible", timeout=10000)
                    await password_locator.click()  # Mimic sync code
                    await password_locator.fill(ncs_password)
                    print("🔒 Filled password using get_by_role('textbox', name='Password').")
                except Exception as e:
                    print(f"⚠️ Failed to fill password with get_by_role('textbox', name='Password'): {e}")
                    # Fallbacks
                    fallbacks = [
                        (page.get_by_role("textbox", name=re.compile(r"Password", re.I)), "password regex"),
                        (page.locator("input[type='password']"), "password input"),
                        (page.locator("input#ctl00_PlaceHolderMain_txtPassword"), "ASP.NET password ID"),
                        (page.locator("input[name='ctl00$PlaceHolderMain$txtPassword']"), "ASP.NET password name")
                    ]
                    for locator, desc in fallbacks:
                        try:
                            if await locator.count() > 0:
                                await locator.first.click()
                                await locator.first.fill(ncs_password)
                                print(f"🔒 Filled password using fallback: {desc}")
                                break
                        except Exception as fallback_e:
                            print(f"⚠️ Fallback {desc} failed: {fallback_e}")

                # Check for iframes
                iframes = page.locator("iframe")
                if await iframes.count() > 0 and not username_filled:
                    print(f"⚠️ Found {await iframes.count()} iframe(s). Attempting to fill inside iframes...")
                    for i in range(await iframes.count()):
                        iframe = iframes.nth(i)
                        frame = await iframe.content_frame()
                        if frame:
                            try:
                                frame_username = frame.get_by_role("textbox", name="User Name")
                                if await frame_username.count() > 0:
                                    await frame_username.click()
                                    await frame_username.fill(ncs_username)
                                    print(f"👤 Filled username in iframe {i}.")
                                    username_filled = True
                                frame_password = frame.get_by_role("textbox", name="Password")
                                if await frame_password.count() > 0:
                                    await frame_password.click()
                                    await frame_password.fill(ncs_password)
                                    print(f"🔒 Filled password in iframe {i}.")
                            except Exception as iframe_e:
                                print(f"⚠️ Iframe {i} fill failed: {iframe_e}")

                if not username_filled:
                    print("❌ Could not locate username field. Please check 'login_page.png' or use Playwright codegen.")
                    await page.screenshot(path="login_no_username.png")
                    print("📸 Saved screenshot to 'login_no_username.png'.")

                # Pause for manual CAPTCHA or verification
                print("⚠️ Check the browser: Verify fields are filled. Complete any CAPTCHA if shown.")
                input("Press Enter when ready to submit the login form...")

                # Click login button
                try:
                    login_button = page.get_by_role("button", name="Sign In")
                    await login_button.wait_for(state="visible", timeout=5000)
                    await login_button.click()
                    print("🔁 Clicked 'Sign In' button.")
                except Exception:
                    # Fallback selectors
                    fallback_buttons = [
                        (page.get_by_role("button", name="Submit"), "Submit button"),
                        (page.get_by_role("button", name="Login"), "Login button"),
                        (page.get_by_role("button", name=re.compile(r"Sign|Login|Submit", re.I)), "regex button"),
                        (page.locator("input[type='submit']"), "submit input"),
                        (page.locator("button[type='submit']"), "submit button")
                    ]
                    clicked = False
                    for locator, desc in fallback_buttons:
                        try:
                            if await locator.count() > 0:
                                await locator.first.click()
                                print(f"🔁 Clicked fallback button: {desc}")
                                clicked = True
                                break
                        except Exception:
                            continue
                    if not clicked:
                        print("❌ Could not click login button. Please click manually in the browser.")
                        await page.screenshot(path="login_no_button.png")
                        print("📸 Saved screenshot to 'login_no_button.png'.")

                # Wait for navigation
                await page.wait_for_timeout(5000)
                current_url = page.url
                if "login" not in current_url.lower() or "dashboard" in current_url.lower() or "home" in current_url.lower():
                    print("✅ Login appears successful (URL changed).")
                else:
                    print("⚠️ Still on login page; check for errors or CAPTCHA.")
                    await page.screenshot(path="login_error.png")
                    print("📸 Saved error screenshot to 'login_error.png'.")

            except PlaywrightTimeoutError as e:
                print(f"❌ Timeout during login navigation or element wait: {e}")
                await page.screenshot(path="login_error.png")
                print("📸 Saved error screenshot to 'login_error.png'.")
                await browser.close()
                return
            except Exception as e:
                print(f"⚠️ General login error: {e}")
                await page.screenshot(path="login_error.png")
                print("📸 Saved error screenshot to 'login_error.png'.")
                await browser.close()
                return
        else:
            print("🔎 Running in guest mode (no login).")
            await page.goto("https://www.ncs.gov.in/Pages/Search.aspx", wait_until="domcontentloaded", timeout=30000)

        # 2) Navigate to Search page
        try:
            current = page.url
            if "Search.aspx" not in current:
                print("➡️ Navigating to NCS search page...")
                await page.goto("https://www.ncs.gov.in/Pages/Search.aspx", wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(1000)
        except Exception as e:
            print(f"⚠️ Issue opening search page: {e}")

        # Apply filters
        print(f"🔎 Applying filters: Sector={sector} | Location={location} | OrgType={org_type} | Qualification={qualification} | JobNature={job_nature}")
        async def try_set_select_by_text(key_text: str) -> bool:
            try:
                res = await page.evaluate(
                    """(text) => {
                        const selects = Array.from(document.querySelectorAll('select'));
                        for (const s of selects) {
                            for (const o of Array.from(s.options)) {
                                if (o.text.toLowerCase().includes(text.toLowerCase())) {
                                    s.value = o.value;
                                    s.dispatchEvent(new Event('change'));
                                    return true;
                                }
                            }
                        }
                        return false;
                    }""",
                    key_text
                )
                return bool(res)
            except Exception:
                return False

        if sector:
            ok = await try_set_select_by_text(sector)
            if ok:
                print("✔️ Sector applied (heuristic).")
            else:
                print("⚠️ Could not automatically apply Sector; you may need to use advanced search URL.")

        if location:
            ok = await try_set_select_by_text(location)
            if ok:
                print("✔️ Location applied (heuristic).")
            else:
                try:
                    inputs = page.locator("input")
                    cnt = await inputs.count()
                    typed = False
                    for i in range(cnt):
                        inp = inputs.nth(i)
                        nm = (await inp.get_attribute("name") or "").lower()
                        ph = (await inp.get_attribute("placeholder") or "").lower()
                        if "location" in nm or "location" in ph or "select location" in ph:
                            await inp.fill(location)
                            typed = True
                            break
                    if typed:
                        print("✔️ Typed Location into an input field (heuristic).")
                    else:
                        print("⚠️ Could not type Location automatically.")
                except Exception:
                    print("⚠️ Error trying to type Location.")

        try:
            btn = page.locator("button:has-text('Search'), input[type='submit'][value*='Search'], button:has-text('Apply')")
            if await btn.count() > 0:
                await btn.first.click()
                print("🔁 Clicked Search / Apply Filters button.")
                await page.wait_for_timeout(1500)
        except Exception:
            pass

        # 3) Scroll to load job cards
        print("📜 Scrolling results to load job listings ...")
        for _ in range(20):
            prev_h = await page.evaluate("document.body.scrollHeight")
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(800)
            new_h = await page.evaluate("document.body.scrollHeight")
            if new_h == prev_h:
                break

        # 4) Collect job card elements
        print("🔗 Collecting job cards ...")
        cards = []
        selectors = [
            "a:has-text('Read more')",
            "a:has-text('Read More')",
            "a:has-text('Read more...')",
            "a:has-text('Apply')",
            "div.job-card",
            ".job-card a",
            ".list-group-item a",
            ".card a"
        ]
        for sel in selectors:
            try:
                loc = page.locator(sel)
                cnt = await loc.count()
                if cnt > 0:
                    for i in range(min(cnt, no_jobs*2)):
                        cards.append(loc.nth(i))
            except Exception:
                continue
            if len(cards) >= no_jobs:
                break

        if len(cards) == 0:
            anchors = await page.locator("a").all()
            for a in anchors:
                try:
                    txt = (await a.inner_text() or "").lower()
                    href = await a.get_attribute("href") or ""
                    if "read" in txt or "apply" in txt or "job" in href.lower() or "job" in txt:
                        cards.append(a)
                        if len(cards) >= no_jobs:
                            break
                except Exception:
                    continue

        print(f"🔎 Found {len(cards)} candidate card elements (will process up to {no_jobs}).")
        processed = 0
        results = {}

        # Process job cards
        for idx, card in enumerate(cards):
            if processed >= no_jobs:
                break
            try:
                print(f"\n🧾 Processing job card #{processed+1} ...")
                raw_text, detail_page = await click_and_extract_job_detail(page, card)
                job_url = detail_page.url if detail_page and hasattr(detail_page, "url") else ""
                parsed = parse_job_text_for_fields(raw_text, job_url)
                job_text_for_sim = "\n".join([parsed.get("title",""), parsed.get("company",""), parsed.get("description","")])
                sim = compare_job_with_resume_text(user_id, job_text_for_sim)
                print(f"   ➤ Similarity with resume: {sim:.3f}")

                results[job_url or f"card_{idx}"] = {
                    "parsed": parsed,
                    "similarity": sim,
                    "cover_letter": None,
                    "apply_decision": None,
                    "applied": False,
                    "detail_page": detail_page
                }

                if sim >= SIMILARITY_THRESHOLD:
                    print("✨ High match detected — auto-opening job details in browser for manual review.")
                    try:
                        if detail_page:
                            await detail_page.bring_to_front()
                            await detail_page.wait_for_timeout(800)
                        summ_res = summarize_text(resume_text, 8)
                        summ_job = summarize_text(parsed.get("description",""), 8)
                        cl = generate_cover_letter(summ_res, summ_job)
                        results[job_url or f"card_{idx}"]["cover_letter"] = cl
                        print("\n--- Cover Letter Preview ---")
                        print(cl[:1200])
                        print("--- end preview ---\n")
                    except Exception as e:
                        print(f"⚠️ Error auto-opening or generating cover letter: {e}")

                    while True:
                        ans = input("👉 Do you want to apply to this job now? (y/n): ").strip().lower()
                        if ans in ("y","yes"):
                            results[job_url or f"card_{idx}"]["apply_decision"] = True
                            results[job_url or f"card_{idx}"]["applied"] = False
                            print("✅ Marked to apply (manual follow-up).")
                            break
                        elif ans in ("n","no"):
                            results[job_url or f"card_{idx}"]["apply_decision"] = False
                            print("⛔ Skipping application for this job.")
                            break
                        else:
                            print("Please enter 'y' or 'n'.")

                else:
                    results[job_url or f"card_{idx}"]["cover_letter"] = "Not a good match"
                    results[job_url or f"card_{idx}"]["apply_decision"] = False
                    print("ℹ️ Not a good match — skipping cover letter generation.")

                processed += 1

                try:
                    if detail_page and detail_page != page:
                        keep_open = results[job_url or f"card_{idx}"]["similarity"] >= SIMILARITY_THRESHOLD and results[job_url or f"card_{idx}"]["apply_decision"] is True
                        if not keep_open:
                            await detail_page.close()
                except Exception:
                    pass

                await page.wait_for_timeout(600)

            except Exception as e:
                print(f"⚠️ Error processing card #{idx+1}: {e}")
                continue

        # Save results to CSV
        print("\n💾 Saving results to CSV ...")
        csv_rows = []
        for key, entry in results.items():
            parsed = entry["parsed"]
            csv_rows.append({
                "url": parsed.get("url") or key,
                "title": parsed.get("title"),
                "company": parsed.get("company"),
                "similarity": entry.get("similarity"),
                "apply_decision": entry.get("apply_decision"),
                "applied": entry.get("applied"),
                "last_date": parsed.get("last_date"),
                "location": parsed.get("location"),
                "salary": parsed.get("salary"),
                "skills": parsed.get("skills"),
                "cover_letter": (entry.get("cover_letter") or "")[:400].replace("\n"," "),
                "description_snippet": parsed.get("description","")[:400].replace("\n"," ")
            })

        csv_file = "ncs_job_results.csv"
        keys = ["url","title","company","similarity","apply_decision","applied","last_date","location","salary","skills","cover_letter","description_snippet"]
        try:
            with open(csv_file, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for r in csv_rows:
                    writer.writerow(r)
            print(f"✅ Results saved to {csv_file}")
        except Exception as e:
            print(f"⚠️ Could not save CSV: {e}")

        # Summary
        print("\n" + "#"*60)
        print("SUMMARY")
        print(f"Total jobs processed: {len(csv_rows)}")
        for i, r in enumerate(csv_rows, 1):
            print(f"\n[{i}] {r['title'] or 'No Title'} @ {r['company'] or 'Unknown'}")
            print(f" URL: {r['url']}")
            print(f" Similarity: {r['similarity']:.3f}")
            print(f" Apply decision: {r['apply_decision']}")
            print(f" Last date: {r['last_date']}")
            print(f" Short desc: {r['description_snippet'][:200]}")
        print("#"*60 + "\n")

        print("🔔 Browser will remain open so you can review any auto-opened job pages.")
        print("When you're done, close the browser window to finish.")
        try:
            input("Press Enter after you finish reviewing and close the browser to terminate the script ...")
        except Exception:
            pass

        try:
            await browser.close()
        except Exception:
            pass

    print("🛑 Agent finished. Check ncs_job_results.csv for details.")


if __name__ == "__main__":
    print("NCS Job Application Agent v2 (visible browser, interactive login + filters)\n")
    choice = input("Do you want to login to NCS with credentials? (y/n) [guest mode if n]: ").strip().lower()
    login_with_creds = choice in ("y", "yes")
    username = password = None
    if login_with_creds:
        username = input("Enter your NCS username/email: ").strip()
        password = getpass.getpass("Enter your NCS password (input hidden): ")

    print("\nEnter Search Filters (leave blank for 'Any'):")
    sector = input("Sector (e.g., IT and Communication, Finance and Insurance): ").strip()
    location = input("Location (e.g., All India, Karnataka, Delhi) or paste filtered NCS URL: ").strip()
    org_type = input("Organisation Type (Government / Private / Both) [default Both]: ").strip() or "Both"
    qualification = input("Minimum Qualification (e.g., Graduate, 12th Pass, Any) [default Any]: ").strip() or "Any"
    job_nature = input("Job Nature (Full Time / Internship / Work From Home / Any) [default Any]: ").strip() or "Any"

    while True:
        try:
            no_jobs = int(input("How many jobs to fetch this run (e.g., 5, 20) [default 10]: ") or 10)
            break
        except ValueError:
            print("Please enter a valid integer.")

    resume_file = input("Enter your resume PDF filename (e.g., resume.pdf) [default 'resume.pdf']: ").strip() or "resume.pdf"
    if not os.path.exists(resume_file):
        print(f"❗ Resume file '{resume_file}' was not found in current directory.")
        resume_file = input("Please provide full path to resume PDF: ").strip()

    print("\nReading resume...")
    resume_text = read_pdf(resume_file)

    # Run agent
    asyncio.run(run_agent_interactive(
        resume_text=resume_text,
        login_with_creds=login_with_creds,
        ncs_username=username,
        ncs_password=password,
        sector=sector,
        location=location,
        org_type=org_type,
        qualification=qualification,
        job_nature=job_nature,
        no_jobs=no_jobs
    ))