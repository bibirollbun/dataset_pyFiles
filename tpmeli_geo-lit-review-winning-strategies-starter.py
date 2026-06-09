# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


%%html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Review: Advanced Deep Learning for FWI (2022â€“2025)</title>

<!-- Link to Prism.js CSS for syntax highlighting -->
<link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-okaidia.min.css" rel="stylesheet" />
<link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/line-numbers/prism-line-numbers.min.css" rel="stylesheet" />

<style>
    /* -------------------------------------------
       Global Resets & Variables
    ------------------------------------------- */
    *, *::before, *::after {
        box-sizing: border-box;
    }

    :root {
        /* Colors - Variables kept, but usage for text/bg removed below */
        --primary-color: #007bff;
        --secondary-color: #0056b3;
        --text-color: #333;             /* Variable defined but not used for base text color below */
        --heading-color: #1a1a1a;       /* Variable defined but not used for heading color below */
        --bg-color: #fcfcfc;            /* Variable defined but not used for body background below */
        --container-bg: #ffffff;        /* Variable defined but not used for container background below */
        --border-color: #e0e0e0;         /* Keep for structural borders */
        --blockquote-bg: #f9f9f9;       /* Variable defined but not used for blockquote background below */
        --blockquote-border: var(--primary-color); /* Keep for accent */
        --code-inline-bg: #f0f0f0;      /* Variable defined but not used for inline code background below */
        --code-block-bg: #272822;        /* Variable defined but not used for code block background below (Prism CSS handles this) */
        --link-color: var(--primary-color); /* Variable defined but not used for link color below */
        --link-hover-color: var(--secondary-color); /* Variable defined but not used for link hover color below */
        --accordion-bg: #f8f9fa;        /* Keep for accordion structure */
        --accordion-border: #dee2e6;    /* Keep for accordion structure */
        --accordion-btn-bg: #e9ecef;    /* Keep for accordion button structure */
        --accordion-btn-hover-bg: #d3d9df;/* Keep for accordion button hover */
        --accordion-btn-active-bg: #ced4da;/* Keep for accordion button active */
        --accordion-icon-color: var(--secondary-color); /* Variable defined but not used for icon color below */
        --copy-btn-bg: #5a5a5a;          /* Keep for button visibility */
        --copy-btn-hover-bg: #777777;    /* Keep for button visibility */
        --copy-btn-active-bg: #4CAF50;   /* Keep for button state */

        /* Shared Design Tokens */
        --border-radius: 6px;         /* Unify border radii across elements */
    }

    /* -------------------------------------------
       Base & Body
    ------------------------------------------- */
    html {
        scroll-behavior: smooth;  /* Smooth scroll for anchor links */
    }

    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif,
                     "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
        line-height: 1.7;
        /* REMOVED: color: var(--text-color); */
        /* REMOVED: background-color: var(--bg-color); */
        /* Browser default text and background colors will now apply */
        margin: 0;
        padding: 0;
        font-size: 16px;
    }

    /* Slightly larger font size on large screens */
    @media (min-width: 1200px) {
        body {
            font-size: 17px;
        }
    }

    /* -------------------------------------------
       Main Container
    ------------------------------------------- */
    .fwi-review-container {
        max-width: 850px;
        margin: 40px auto; /* Increased spacing top/bottom */
        padding: 30px;
        /* REMOVED: background-color: var(--container-bg); */ /* Container will be transparent to body background */
        border: 1px solid var(--border-color); /* Keep border for structure */
        border-radius: var(--border-radius); /* Unified border radius */
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06); /* Keep shadow for depth */
    }

    /* -------------------------------------------
       Headings
    ------------------------------------------- */
    h1, h2, h3, h4 {
        /* REMOVED: color: var(--heading-color); */ /* Will inherit browser default text color */
        margin-top: 2.2em; /* Base increased spacing */
        margin-bottom: 1em;
        line-height: 1.4;
        font-weight: 600;
        font-family: Georgia, serif;
    }

    h1 {
        font-size: 2.2em;
        text-align: center;
        border-bottom: 3px solid var(--primary-color); /* Keep accent border */
        padding-bottom: 0.5em;
        margin-top: 0; /* First element */
        margin-bottom: 1.5em;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.05); /* Keep subtle shadow */
    }

    h2 {
        font-size: 1.8em;
        border-left: 5px solid var(--primary-color); /* Keep accent border */
        padding-left: 0.6em;
        margin-top: 3.2em; /* Increased padding */
        margin-bottom: 1em; /* Increased spacing below */
    }

    h3 {
        font-size: 1.4em;
        /* REMOVED: color: var(--secondary-color); */ /* Will inherit browser default text color */
        margin-top: 2.8em; /* Increased padding */
    }

    h4 {
        font-size: 1.15em;
        font-style: normal;
        /* REMOVED: color: #444; */ /* Will inherit browser default text color */
        margin-top: 2.4em; /* Increased padding */
        margin-bottom: 0.8em;
        font-weight: 600;
    }

    /* -------------------------------------------
       Paragraphs
    ------------------------------------------- */
    p {
        margin-bottom: 1.5em;
        text-align: left;
        /* Inherits text color from body */
    }

    /* -------------------------------------------
       Links
    ------------------------------------------- */
    a {
        /* REMOVED: color: var(--link-color); */ /* Use browser default link color */
        text-decoration: none;
        transition: color 0.2s ease, border-bottom 0.2s ease;
        border-bottom: none; /* Subtle by default */
    }
    a:hover, a:focus {
        /* REMOVED: color: var(--link-hover-color); */ /* Use browser default link hover color */
        border-bottom: 1px solid currentColor; /* Use current text color for underline on hover */
    }

    /* -------------------------------------------
       Lists
    ------------------------------------------- */
    ul, ol {
        padding-left: 30px;
        margin-bottom: 1.5em;
    }
    li {
        margin-bottom: 0.8em;
    }
    ul ul, ol ol, ul ol, ol ul {
        margin-top: 0.5em;
        margin-bottom: 0.8em;
    }

    /* -------------------------------------------
       Inline Code
    ------------------------------------------- */
    code:not([class*="language-"]) {
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
        /* REMOVED: background-color: var(--code-inline-bg); */
        padding: 0.2em 0.5em;
        border-radius: 4px;
        font-size: 0.9em;
        /* REMOVED: color: #a10606; */ /* Use default text color */
        white-space: nowrap;
    }

    /* -------------------------------------------
       Code Blocks & Copy Button
       (Prism.js theme primarily handles syntax)
    ------------------------------------------- */
    pre {
        /* --- BASE PRE STYLES --- */
        border-radius: var(--border-radius);
        margin-bottom: 1.8em; /* Default margin for standalone pre */
        border: none;
        position: relative; /* For copy button positioning */
        /* Rely on linked Prism theme for background */
        /* Base overflow settings (can be overridden by .code-container pre) */
        overflow: auto;
    }
    pre code {
        display: block; /* Ensure code fills pre for correct scroll */
        font-size: 0.9em;
        line-height: 1.5;
        /* Rely on linked Prism theme for text/bg colors */
        padding: 1em; /* Standard padding inside code block */
        white-space: pre; /* Preserve whitespace and allow scrolling */
    }

    .copy-code-btn {
        position: absolute;
        top: 8px;
        right: 8px;
        z-index: 10;
        padding: 5px 8px;
        font-size: 0.75em;
        color: white; /* Keep for button visibility */
        background-color: var(--copy-btn-bg); /* Keep for button visibility */
        border: none;
        border-radius: 4px;
        cursor: pointer;
        opacity: 0.7;
        transition: opacity 0.2s ease, background-color 0.2s ease;
    }
    .copy-code-btn:hover {
        opacity: 1;
        background-color: var(--copy-btn-hover-bg); /* Keep for button visibility */
    }
    .copy-code-btn:active {
        opacity: 1;
    }
    .copy-code-btn.copied {
        background-color: var(--copy-btn-active-bg); /* Keep for button state */
        color: white; /* Keep for button state */
    }
    /* Hide copy button if JS fails or isn't needed */
    .no-js .copy-code-btn {
        display: none;
    }
    /* Hide copy button when container is collapsed */
    .code-collapsed .copy-code-btn {
        display: none;
    }

    /* -------------------------------------------
       Blockquotes
    ------------------------------------------- */
    blockquote {
        border-left: 5px solid var(--blockquote-border); /* Keep accent border */
        margin: 2em 0;
        padding: 1.2em 1.6em;
        /* REMOVED: background-color: var(--blockquote-bg); */
        /* REMOVED: color: #555; */ /* Inherit default text color */
        font-style: italic;
        border-radius: 0 var(--border-radius) var(--border-radius) 0;
    }
    blockquote p {
        margin-bottom: 0.5em;
    }

    /* -------------------------------------------
       Sources Section
    ------------------------------------------- */
    .sources ul {
        list-style: none;
        padding-left: 0;
    }
    .sources li {
        margin-bottom: 1em;
        font-size: 0.95em;
    }
    .sources a {
        font-weight: 500;
        /* Inherits link color */
    }

    /* -------------------------------------------
       Code Toggle Button & Container (REFINED & SCROLLABLE)
    ------------------------------------------- */
    .toggle-code-btn {
        display: inline-block;
        background-color: var(--primary-color); /* Keep button color */
        color: white; /* Keep button text color */
        border: none;
        padding: 5px 10px;
        font-size: 0.8em;
        border-radius: var(--border-radius);
        cursor: pointer;
        margin-bottom: 8px;
        transition: background-color 0.2s ease;
    }
    .toggle-code-btn:hover {
        background-color: var(--secondary-color); /* Keep button hover color */
    }
    .code-container {
        margin-bottom: 1.8em;
    }
    .code-container pre {
        margin-bottom: 0; /* Remove bottom margin when inside container */
        transition: max-height 0.3s ease-out, opacity 0.2s ease-out, padding-top 0.3s ease-out, padding-bottom 0.3s ease-out;
        /* --- KEY CHANGES FOR CODE BLOCK SCROLLING --- */
        /* 1. Set a fixed max-height. Adjust this value (e.g., 600px, 70vh) as needed. */
        max-height: 600px;
        /* 2. Enable vertical scrolling when content exceeds max-height */
        overflow-y: auto;
        /* 3. Enable horizontal scrolling for long lines (important for <pre>) */
        overflow-x: auto;
        /* --- END KEY CHANGES --- */
        opacity: 1;
        padding: 1em; /* Ensure padding is present when visible */
        /* Ensure white-space behavior is correct for pre */
        white-space: pre;
    }
    .code-collapsed pre {
        max-height: 0;
        opacity: 0;
        margin-bottom: 0;
        padding-top: 0; /* Remove padding when collapsed */
        padding-bottom: 0;
        border: none;
        overflow: hidden; /* Explicitly hide overflow when collapsed for transition */
    }

    /* -------------------------------------------
       Utility Class for Emphasis
    ------------------------------------------- */
    .key-concept {
        font-weight: 600;
        color: var(--secondary-color); /* Keep specific color for emphasis */
        background-color: #fffceb; /* Keep specific background for emphasis */
        padding: 0.2em 0.4em;
        border-radius: 4px;
    }

    /* -------------------------------------------
       Accordion Styles (REFINED & SCROLLABLE)
    ------------------------------------------- */
    .accordion-item {
        background-color: var(--accordion-bg); /* Keep bg for structure */
        border: 1px solid var(--accordion-border); /* Keep border for structure */
        border-radius: var(--border-radius);
        margin-bottom: 1em;
        overflow: hidden; /* Contain border radius and content */
    }
    .accordion-button {
        background-color: var(--accordion-btn-bg); /* Keep bg for structure */
        /* REMOVED: color: #222; */ /* Inherit text color */
        cursor: pointer;
        padding: 12px 18px;
        width: 100%;
        border: none;
        text-align: left;
        outline: none;
        font-size: 1em;
        font-weight: 600;
        transition: background-color 0.3s ease;
        position: relative;
    }
    .accordion-button:hover {
        background-color: var(--accordion-btn-hover-bg); /* Keep bg for structure */
    }
    .accordion-button:focus {
        outline: 2px dashed var(--primary-color); /* Keep focus style */
        outline-offset: 2px;
    }
    .accordion-button.active {
        background-color: var(--accordion-btn-active-bg); /* Keep bg for structure */
    }
    .accordion-button::after {
        content: '+';
        font-size: 1.2em;
        /* REMOVED: color: var(--accordion-icon-color); */ /* Inherit text color */
        position: absolute;
        right: 18px;
        top: 50%;
        transform: translateY(-50%);
        transition: transform 0.2s ease-in-out;
    }
    .accordion-button.active::after {
        content: 'âˆ’';
        transform: translateY(-50%);
    }

    .accordion-content {
        padding: 0 18px; /* Horizontal padding */
        /* REMOVED: background-color: #fff; */ /* Let it inherit */
        max-height: 0; /* Start collapsed */
        overflow: hidden; /* Hide content and prevent scrollbars when collapsed */
        opacity: 0;
        transition: max-height 0.35s ease-in-out, opacity 0.3s ease-in-out, padding-top 0.35s ease-in-out, padding-bottom 0.35s ease-in-out;
        font-size: 0.95em;
        line-height: 1.6;
        padding-top: 0;
        padding-bottom: 0;
        border-top: none;
    }

    /* --- MODIFICATION FOR ACCORDION SCROLLING --- */
    .accordion-content.show {
        opacity: 1;
        /* --- KEY CHANGES --- */
        /* 1. Set a fixed max-height. Adjust this value (e.g., 300px, 50vh) as needed. */
        max-height: 400px;
        /* 2. Enable vertical scrolling ONLY when content exceeds the fixed max-height */
        overflow-y: auto;
        /* --- END KEY CHANGES --- */

        padding-top: 1.5em; /* Add padding when shown */
        padding-bottom: 1.5em; /* Add padding when shown */
        border-top: 1px solid var(--accordion-border); /* Add separator only when open */
    }
    /* --- END MODIFICATION --- */


    /* Adjust margins of first/last elements inside accordion content */
    .accordion-content > *:first-child {
        margin-top: 0;
    }
    .accordion-content > *:last-child {
        margin-bottom: 0;
    }

    .accordion-content h5 {
        font-size: 1.05em;
        /* REMOVED: color: var(--secondary-color); */ /* Inherit text color */
        margin-top: 1.8em;
        margin-bottom: 0.6em;
        font-weight: 600;
        border-bottom: 1px dotted var(--border-color);
        padding-bottom: 0.2em;
    }
    /* Add some space above h5 if it's not the first element */
    .accordion-content > *:not(:first-child) + h5 {
         margin-top: 2.0em;
    }


    /* -------------------------------------------
       Math & Subscripts
    ------------------------------------------- */
    .math-notation {
        font-family: 'Computer Modern', 'Latin Modern Math', serif;
        font-style: italic;
        /* Color will inherit */
    }
    .math-symbol {
        font-weight: bold;
        /* Color will inherit */
    }
    .subscript {
        vertical-align: sub;
        font-size: smaller;
    }
    .superscript {
        vertical-align: super;
        font-size: smaller;
    }

    /* -------------------------------------------
       Podcast Link Emoji
    ------------------------------------------- */
    .podcast-link::before {
        content: "ğŸ�µ ";
        display: inline;
    }

    /* -------------------------------------------
       Previous Competitions Styling
    ------------------------------------------- */
    .previous-competitions h3 {
        font-size: 1.25em;
        /* REMOVED: color: var(--primary-color); */ /* Inherit text color */
        margin-top: 2.5em; /* Increased padding */
        margin-bottom: 0.8em;
        border-bottom: 1px dotted var(--border-color);
        padding-bottom: 0.3em;
    }
    .previous-competitions ul {
        padding-left: 25px;
        list-style-type: disc;
    }
    .previous-competitions li {
        margin-bottom: 0.6em;
        font-size: 0.95em;
    }

    /* -------------------------------------------
       Table Styling (If needed)
    ------------------------------------------- */
    table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 1.5em;
    }
    th, td {
        border: 1px solid var(--border-color);
        padding: 0.75em 1em;
        /* Inherit text color */
    }
    th {
        background-color: #f2f2f2; /* Keep background for table header structure */
        font-weight: 600;
        /* Inherit text color */
    }
</style>

</head>
<body>

<div class="fwi-review-container" id="fwi-review-cell-content-123"> <!-- Use a unique ID -->

    <div id="main-content">

        <h2 id="intro">Introduction: Seeing Beneath the Surface with Sound and AI</h2>

        <p class="podcast-link" style="text-align: center; margin-bottom: 2em;">
            Podcast for this entire notebook:
            <a href="https://notebooklm.google.com/notebook/e87b0226-446d-486a-8870-2c218390b427/audio" target="_blank" rel="noopener">
                Listen to the audio podcast of this notebook.
            </a>
        </p>

        <p class="podcast-link" style="text-align: center; margin-bottom: 2em;">
            Podcast of Insights From Previous Related Competitions:
            <a href="https://notebooklm.google.com/notebook/dcbaa999-61ef-4a05-9f33-865f22acb24f/audio" target="_blank" rel="noopener">
                Podcast of Insights From Previous Related Competitions
            </a>
        </p>


        <img src="https://www.thomasmeli.com/images/kaggle/geo-fwi-2025/prediction.webp" alt="FWI Prediction Example" style="max-width: 100%; height: auto; display: block; margin: 1em auto;">

        <p>
            Imagine trying to understand the complex layers inside a cakeâ€”sponge, cream, fruitâ€”without slicing it open. How could you map its internal structure? Geoscientists face a similar challenge when mapping the Earth's subsurface, looking for features like rock layers, oil and gas reservoirs, or geological faults. <strong class="key-concept">Full Waveform Inversion (FWI)</strong> is a sophisticated geophysical technique designed to tackle this.
        </p>

        <div class="accordion-item">
            <button class="accordion-button">What is Full Waveform Inversion (FWI)?</button>
            <div class="accordion-content">
                <p>Think of FWI like a highly detailed ultrasound or sonar for the Earth. We send sound (seismic) waves into the ground using sources like vibrating trucks or small, controlled explosions. These waves travel down, bounce off different underground layers and structures, and return to the surface where they are recorded by many sensors (geophones or hydrophones). FWI uses the *entire* recording of these returning waves (the "full waveform," including all wiggles and timing details) to build a detailed map of how fast the sound waves travel through different parts of the subsurface. This map, called a velocity model, reveals the underground structure.</p>

                <h5>More Rigorous Definition</h5>
                <p>FWI is a high-resolution geophysical imaging technique that aims to estimate quantitative models of subsurface physical parameters (primarily seismic velocities, but potentially density, attenuation, or anisotropy) by minimizing the mismatch between observed seismic waveform data and synthetic waveform data generated by numerically solving the wave equation using a candidate subsurface model. It utilizes the complete information content of the recorded seismograms (amplitude, phase, travel time).</p>

                <h5>Mathematical Intuition</h5>
                <p>The core idea is to minimize an objective (or cost) function, often the L2 norm (least-squares) difference between observed and synthetic data:</p>
                <p><span class="math-notation">Î¦(<span class="math-notation">v</span>) = Â½ Î£<sub class="subscript">r,t</sub> [ <span class="math-notation">u</span><sub class="subscript">obs</sub>(<span class="math-notation">r, t</span>) - <span class="math-notation">u</span><sub class="subscript">syn</sub>(<span class="math-notation">r, t</span>; <span class="math-notation">v</span>) ]Â²</span></p>
                <p>Where:</p>
                <ul>
                    <li><span class="math-notation">Î¦(<span class="math-notation">v</span>)</span> is the misfit (error) for a given velocity model <span class="math-notation">v</span>.</li>
                    <li><span class="math-notation">v(<span class="math-notation">x</span>)</span> is the velocity model (the unknown parameter we want to find, defined at spatial location <span class="math-notation">x</span>).</li>
                    <li><span class="math-notation">u</span><sub class="subscript">obs</sub><span class="math-notation">(r, t)</span> is the observed seismic data recorded at receiver <span class="math-notation">r</span> and time <span class="math-notation">t</span>.</li>
                    <li><span class="math-notation">u</span><sub class="subscript">syn</sub><span class="math-notation">(r, t; <span class="math-notation">v</span>)</span> is the synthetic seismic data simulated using the current velocity model <span class="math-notation">v</span>.</li>
                </ul>
                <p>The synthetic data <span class="math-notation">u</span><sub class="subscript">syn</sub> is calculated by solving the wave equation (often the acoustic approximation for simplicity, though elastic is more realistic):</p>
                <p><span class="math-notation">âˆ‚Â²<span class="math-notation">u</span>/âˆ‚tÂ² = <span class="math-notation">v</span>(<span class="math-notation">x</span>)Â² âˆ‡Â²<span class="math-notation">u</span> + <span class="math-notation">s</span>(<span class="math-notation">x, t</span>)</span></p>
                <p>Where <span class="math-notation">u(x,t)</span> is the pressure wavefield, <span class="math-notation">âˆ‡Â²</span> is the Laplacian operator (representing spatial derivatives), and <span class="math-notation">s(x,t)</span> is the seismic source. FWI iteratively updates <span class="math-notation">v</span> to minimize <span class="math-notation">Î¦</span>, typically using gradient-based optimization methods like gradient descent or L-BFGS.</p>

                 <h5>FAQ: Why is FWI important?</h5>
                 <p>FWI aims for higher resolution imaging compared to simpler methods like travel-time tomography. This detail is crucial for applications like:</p>
                 <ul>
                    <li>Identifying oil and gas reservoirs or geothermal resources.</li>
                    <li>Mapping subsurface structures for carbon sequestration or storage.</li>
                    <li>Assessing geological hazards (e.g., fault structures).</li>
                    <li>Guiding civil engineering projects (e.g., tunnel construction).</li>
                    <li>Understanding fundamental Earth structure.</li>
                 </ul>
            </div>
        </div>

        <p>
            Think of FWI like performing an ultrasound on the Earth. We generate seismic waves (using controlled sources like vibrations or small explosions) that travel down into the ground. These waves bounce off different materials and structures, and the returning echoesâ€”the full <span class="key-concept">waveforms</span>â€”are recorded by sensors (geophones or hydrophones) at the surface. FWI analyzes the intricate details of these recorded waveforms to reconstruct a map of the subsurface, specifically a <strong class="key-concept">"velocity model"</strong>. This model shows how fast seismic waves travel through different regions; for example, waves travel faster through dense rock and slower through softer sediments or fluids like water or oil.
        </p>

        <div class="accordion-item">
            <button class="accordion-button">What are Waveforms and Velocity Models?</button>
            <div class="accordion-content">
                <p><strong>Waveform:</strong> Imagine dropping a pebble in a pond. The ripples spreading out are like seismic waves. A waveform is the detailed recording of how the water level (or ground motion) goes up and down over time at a specific sensor location as those ripples pass by. It contains information about the strength, timing, and shape of the waves.</p>
                <p><strong>Velocity Model:</strong> This is the map FWI tries to create. It shows the speed of sound (seismic waves) at every point underground. Just like light travels slower through water than air, seismic waves travel at different speeds through different materials (rock, sand, water, oil). A velocity model helps us "see" these different materials and structures.</p>

                <h5>More Rigorous Definition</h5>
                <p><strong>Waveform:</strong> A seismogram or trace representing the time series of ground motion (e.g., displacement, velocity, acceleration, or pressure for acoustic waves) recorded by a single seismic sensor (receiver) resulting from a seismic source excitation. The "full" waveform includes all recorded variations, including amplitude, phase, and frequency content, over the recording duration.</p>
                <p><strong>Velocity Model:</strong> A spatial distribution (2D or 3D) of seismic wave propagation speeds within the subsurface volume of interest. Typically, this refers to P-wave (compressional) velocity (<span class="math-notation">Vp</span>) and/or S-wave (shear) velocity (<span class="math-notation">Vs</span>). It is the primary parameter estimated in most FWI applications, as it strongly controls wave travel times and reflection/transmission amplitudes.</p>
            </div>
        </div>


        <p>
            Traditionally, FWI relies on iteratively solving complex wave physics equations. This process is computationally intensive, akin to solving a massive, interconnected puzzle where every piece affects every other. Furthermore, traditional methods can easily get stuck on incorrect solutions if the initial guess of the velocity model isn't close enough to the true modelâ€”a notorious problem known as <strong class="key-concept">"cycle-skipping"</strong>.
        </p>

        <div class="accordion-item">
            <button class="accordion-button">What is Cycle-Skipping and why is it a problem?</button>
            <div class="accordion-content">
                <p>Imagine trying to match two identical songs played slightly out of sync. If they are only slightly off, it's easy to tell which way to shift one to match the other. But if one song is delayed by, say, half its length, you might mistakenly try to align the start of the delayed song with the middle of the original one â€“ you've skipped a whole "cycle." In FWI, if our initial guess for the velocity model is too far off, the simulated waves might arrive much earlier or later than the real waves (by more than half a wavelength). The algorithm then gets confused about *which* wiggle in the simulated data should correspond to a wiggle in the real data and makes the wrong adjustment, often getting stuck in a bad solution far from the truth.</p>

                <h5>More Rigorous Definition</h5>
                <p>Cycle-skipping occurs in FWI when the phase difference between the observed (<span class="math-notation">u</span><sub class="subscript">obs</sub>) and synthetic (<span class="math-notation">u</span><sub class="subscript">syn</sub>) waveforms exceeds half a period (Ï€ radians or 180 degrees) for the dominant frequency being considered. Standard local optimization methods (like gradient descent) rely on minimizing the difference based on the nearest corresponding phase cycle. When cycle-skipped, the computed gradient points in a direction that increases, rather than decreases, the true phase difference, leading the inversion to converge towards a local minimum of the objective function <span class="math-notation">Î¦</span> that does not correspond to the true Earth model.</p>
                <p>This typically happens when the initial velocity model is inaccurate, especially for higher frequencies or larger velocity errors, making FWI highly sensitive to the starting model quality.</p>

                 <h5>FAQ: How do advanced methods try to mitigate cycle-skipping?</h5>
                 <p>Several strategies are employed:</p>
                 <ul>
                    <li><strong>Multi-scale approaches:</strong> Start FWI with low frequencies (longer wavelengths, less prone to skipping) and gradually introduce higher frequencies.</li>
                    <li><strong>Improved objective functions:</strong> Use misfit measures less sensitive to phase shifts (e.g., based on wavefield envelopes, optimal transport, or correlation).</li>
                    <li><strong>Better starting models:</strong> Use other methods (like tomography) to get a better initial guess.</li>
                    <li><strong>Deep Learning methods:</strong> Some DL approaches, like IFWI (Implicit FWI) or certain GAN formulations (FWI-GAN), seem inherently more robust to cycle-skipping, potentially because they explore the solution space differently or implicitly incorporate broader structural constraints learned from data or physics.</li>
                 </ul>
            </div>
        </div>


        <blockquote>
            <p>Recently, the field has turned to <strong class="key-concept">Deep Learning (DL)</strong>.</p>
            <p>In the context of FWI, we train a neural network by showing it many examples of seismic data (the "echoes") and the corresponding correct subsurface velocity maps. Early DL models, such as <code>InversionNet</code> (often based on a <span class="key-concept">U-Net architecture</span>, popular in image analysis for its ability to capture details at different scales) and <code>VelocityGAN</code> (using <span class="key-concept">Generative Adversarial Networks</span>, where two networks compete to generate realistic results), demonstrated that AI could learn to predict the subsurface map directly from seismic data, often much faster than traditional methods.</p>
        </blockquote>

        <div class="accordion-item">
            <button class="accordion-button">What are Deep Learning, U-Nets, and GANs in this context?</button>
            <div class="accordion-content">
                <p><strong>Deep Learning (DL):</strong> A type of AI that uses interconnected "neurons" in layered structures (like a brain) to learn complex patterns from data. We "teach" it by showing many examples (e.g., seismic recordings and the matching underground maps). After training, it can often make predictions on new data very quickly.</p>
                <p><strong>U-Net Architecture:</strong> A specific type of DL network design, originally popular for medical image analysis. It looks like the letter 'U'. It first shrinks the input image down (capturing broad features) and then expands it back up (reconstructing fine details), while also using "shortcuts" to carry information from the shrinking path directly to the expanding path. This helps it create detailed output maps (like velocity models) that preserve spatial information well.</p>
                <p><strong>Generative Adversarial Networks (GANs):</strong> Imagine an art forger (Generator) trying to create fake paintings and an art detective (Discriminator) trying to spot the fakes. The forger learns by getting feedback from the detective. In FWI, the Generator tries to create realistic velocity models (or the seismic data resulting from them), and the Discriminator tries to tell the difference between the Generator's output and real-world examples. This competition forces the Generator to become very good at creating realistic results.</p>

                <h5>More Rigorous Definition</h5>
                <p><strong>Deep Learning (DL):</strong> A subfield of machine learning based on artificial neural networks with multiple layers (deep architectures) between the input and output layers. These networks learn hierarchical representations of data, enabling them to model complex, non-linear relationships. In FWI, DL is often used for supervised learning (mapping seismic data to velocity models), unsupervised learning (learning data distributions, e.g., with GANs), or physics-informed learning (integrating physical laws).</p>
                <p><strong>U-Net Architecture:</strong> A convolutional neural network (CNN) architecture characterized by an encoder path (down-sampling, feature extraction) and a symmetric decoder path (up-sampling, localization), with skip connections linking corresponding layers between the encoder and decoder. These skip connections allow the network to combine high-level semantic features (from deeper layers) with low-level fine-grained detail (from earlier layers), making it effective for image segmentation and image-to-image translation tasks, such as mapping seismic data to velocity maps.</p>
                <p><strong>Generative Adversarial Networks (GANs):</strong> A class of deep learning frameworks where two neural networks, a Generator (G) and a Discriminator (D), are trained simultaneously in opposition. The Generator learns to map from a latent space (e.g., random noise) to a target data distribution (e.g., realistic velocity models or seismic data), while the Discriminator learns to distinguish between real data samples and samples generated by G. The training objective is typically formulated as a minimax game. Variations like Wasserstein GAN (WGAN) improve training stability and correlation with sample quality.</p>
            </div>
        </div>


        <p>
            However, these pioneering AI models often struggled with <strong class="key-concept">generalization</strong>. Like a student who only memorizes answers from one textbook, they performed well on geology similar to their training data but failed when faced with new, unseen subsurface structures. To address this, the research community developed large-scale, diverse benchmark datasets, notably the <strong class="key-concept">OpenFWI dataset</strong>. OpenFWI provides a rich "library" of simulated seismic data and corresponding velocity models covering various geological complexities: simple layered structures ("Vel"), layers disrupted by faults ("Fault"), and highly variable, complex geology ("Style"). These datasets allow for more robust training and fairer comparison of different DL approaches.
        </p>

        <div class="accordion-item">
            <button class="accordion-button">What are Generalization and the OpenFWI dataset?</button>
            <div class="accordion-content">
                <p><strong>Generalization:</strong> This is the AI model's ability to perform well on new, unseen data that it wasn't explicitly trained on. A model with good generalization is like a student who understands the underlying concepts and can solve new types of problems, not just the ones they practiced.</p>
                <p><strong>OpenFWI Dataset:</strong> Think of this as a large, diverse collection of practice problems for FWI AI models. It contains thousands of simulated examples of seismic recordings paired with the known "answer key" (the true underground velocity map). Crucially, it includes different types of geological structures (simple layers, faulted layers, complex styles) so that models trained on it learn to handle a wider variety of situations and generalize better.</p>

                <h5>More Rigorous Definition</h5>
                <p><strong>Generalization:</strong> In machine learning, generalization refers to a model's ability to adapt properly to new, previously unseen data drawn from the same distribution as the one used to train the model. A model generalizes well if its performance on a held-out test dataset is close to its performance on the training dataset. Poor generalization (overfitting) occurs when a model learns the training data too specifically, including noise and spurious correlations, and fails to capture the underlying patterns applicable to new data.</p>
                <p><strong>OpenFWI Dataset:</strong> A suite of large-scale, open-source benchmark datasets designed specifically for training and evaluating deep learning-based FWI methods. It comprises multiple families (e.g., FlatVel, CurveVel, FaultSet, StyleSet) containing hundreds of thousands of pairs of 2D synthetic velocity models and their corresponding simulated seismic shot gathers. The diversity in geological structures and acquisition geometries within OpenFWI facilitates the development and comparison of models with improved generalization capabilities. (Reference: Deng et al., 2022; Huang et al., 2022)</p>
            </div>
        </div>

        <h2 id="recent-advances">Recent Advances (2022â€“2025): Smarter AI for Better Underground Maps</h2>

        <p>
            Research from 2022 to 2025 has focused intensely on overcoming the limitations of early DL-based FWI, leading to several key trends:
        </p>

        <ul>
            <li>
                <strong>Bigger Brains, More Experience (Larger Models & Datasets):</strong> Just as humans learn better with more diverse experiences, DL models benefit significantly from training on vast datasets. Jin et al. (2024) demonstrated that training a very large model (<code>BigFWI</code>) on a massive combination of OpenFWI datasets (470,000 examples!) dramatically improved accuracy and generalization ability, reducing errors by 13â€“20%. They also confirmed that increasing model size (more parameters or "neurons") further enhances performance. This underscores the importance of both data scale and model capacity.
                <span class="podcast-link"><a href="https://notebooklm.google.com/notebook/f4f53cb4-42dc-4b7f-95c7-478d750d20a7/audio" target="_blank" rel="noopener">Big Data Impact Podcast</a></span>
            </li>

            <li>
                <strong>Seeing the Big Picture (Transformer Architectures):</strong> Traditional image processing models like <span class="key-concept">Convolutional Neural Networks (CNNs)</span> examine data through small, sliding windows (kernels). This is effective for local patterns but can miss long-range dependencies in the data. Imagine trying to understand a complex musical piece by only listening to two notes at a time. <strong class="key-concept">Transformers</strong>, an architecture that revolutionized natural language processing, employ a mechanism called <strong class="key-concept">"self-attention."</strong>

                <div class="accordion-item">
                    <button class="accordion-button">What are CNNs, Transformers, and Self-Attention?</button>
                    <div class="accordion-content">
                        <p><strong>Convolutional Neural Networks (CNNs):</strong> These AI models are inspired by human vision. They use small "filters" (like magnifying glasses) that slide across the input (like an image or seismic data) to detect local patterns (edges, textures). They are very good at tasks where local features are important, but might struggle to connect information that is far apart in the input.</p>
                        <p><strong>Transformers:</strong> A newer type of AI architecture, originally famous for understanding language (like translating sentences). Instead of just looking locally, Transformers use a special mechanism called "self-attention" to look at *all* parts of the input data simultaneously and figure out how relevant each part is to every other part. This allows them to capture long-range connections and understand the global context.</p>
                        <p><strong>Self-Attention:</strong> Think of it as the model asking, for each piece of input data (e.g., a specific seismic echo), "Which other echoes, no matter how far away in time or space, are most important for understanding *this* echo?" It calculates "attention scores" between all pairs of input pieces, allowing it to weigh the influence of different parts dynamically based on the context.</p>

                        <h5>More Rigorous Definition</h5>
                        <p><strong>Convolutional Neural Networks (CNNs):</strong> A class of deep neural networks highly effective for processing grid-like data (e.g., images, seismic sections). They employ convolutional layers that apply learnable filters (kernels) across the input, leveraging parameter sharing and spatial hierarchies to extract features. Their strength lies in capturing local spatial patterns and building translation-equivariant representations.</p>
                        <p><strong>Transformers:</strong> A neural network architecture, introduced by Vaswani et al. (2017), that relies primarily on self-attention mechanisms instead of recurrence (like RNNs) or local convolution (like CNNs). Originally designed for sequence-to-sequence tasks in NLP, they have been adapted for vision (ViT) and other domains. Their ability to model long-range dependencies makes them suitable for tasks where global context is important.</p>
                        <p><strong>Self-Attention:</strong> A mechanism within the Transformer architecture that allows the model to weigh the importance of different positions in the input sequence when computing the representation of a specific position. It calculates attention scores based on pairwise similarity (often using scaled dot-product attention) between query, key, and value vectors derived from the input embeddings. Multi-head attention allows the model to attend to information from different representational subspaces simultaneously.</p>
                    </div>
                </div>

                <blockquote>
                    Think of self-attention as enabling the model to weigh the importance of all input parts relative to each other, simultaneously. For seismic data, this means it can correlate an early echo at one sensor with a related, delayed echo at a far-off sensor, capturing the 'global context' of the wavefield. Models like the <strong class="key-concept">Seismic Velocity Inversion Transformer (SVIT)</strong> leverage this global view to produce more accurate and geologically plausible velocity maps. We'll discuss SVIT in more detail below.
                </blockquote>
            </li>

            <li>
                <strong>Teaching AI Physics (Hybrid & Physics-Informed Learning):</strong> Instead of solely learning patterns from data examples (input seismic -> output velocity map), newer methods integrate the underlying physics of wave propagation directly into the AI's learning process.

                <div class="accordion-item">
                    <button class="accordion-button">What is Physics-Informed Learning / PINNs?</button>
                    <div class="accordion-content">
                        <p>Imagine training an AI apprentice not just by showing them examples, but also by giving them the physics textbook. Physics-informed methods embed the known laws of physics (like the wave equation in FWI) into the AI's learning process. This can be done by making the AI's predictions satisfy the physics equations, or by using a physics simulator as part of the AI training loop. The goal is to get predictions that are not only consistent with the observed data but also physically plausible.</p>

                        <h5>More Rigorous Definition</h5>
                        <p><strong>Physics-Informed Neural Networks (PINNs):</strong> A class of neural networks trained to solve supervised learning tasks while respecting constraints imposed by physical laws, typically described by partial differential equations (PDEs). The loss function includes not only the data mismatch term (comparing predictions to observations) but also a term that measures how well the network's output satisfies the governing PDE(s), boundary conditions, and initial conditions. This encourages the network to learn solutions that are physically consistent, potentially improving generalization and reducing the need for large labeled datasets.</p>
                        <p>Hybrid approaches might involve using DL for parts of the FWI workflow (e.g., initial model building, denoising) or embedding a differentiable physics simulator within a larger DL framework (like FWI-GAN or IFWI).</p>
                    </div>
                </div>

                <ul>
                    <li><strong>FWI-GAN:</strong> Uses a <span class="key-concept">Generative Adversarial Network (GAN)</span> framework. A "Generator" network proposes a velocity map. A physics simulator calculates the seismic data that *would* result from this map. A second "Discriminator" network learns to distinguish between this simulated data and the real, observed seismic data. The Generator's goal is to fool the Discriminator, forcing it to learn velocity maps that are consistent with both the observed data and the wave physics. Remarkably, this can work in an <span class="key-concept">unsupervised</span> manner, without needing pre-existing "correct" velocity maps for training!</li>
                    <li><strong>Implicit FWI (IFWI):</strong> Represents the velocity map not as a grid of pixels, but as a continuous mathematical function defined by a neural network (often a simple <span class="key-concept">Multi-Layer Perceptron, MLP</span>). You give the network coordinates (x, z), and it outputs the velocity at that exact point. This network's parameters are optimized by comparing the seismic data simulated using its function (via a physics engine) against the real observed data. This approach inherently respects the continuous nature of physical fields and shows promise in avoiding the cycle-skipping problem.</li>
                </ul>
                <p>These physics-informed methods aim to ensure the AI's predictions adhere to physical laws, leading to more trustworthy and reliable results.</p>
            </li>

             <li>
                <strong>Specialists Working Together (Two-Stage & Multi-Task Networks):</strong> Sometimes, a single AI model struggles to excel at all aspects of the FWI problem. For instance, standard models often blur sharp geological boundaries. The <strong class="key-concept">Velocity-Interface Fusion (VIF) Network</strong> (2024) employs a team strategy. One network estimates a smooth velocity field, while a parallel network specializes in identifying the locations of sharp interfaces (like edges in an image). A third network then fuses these two pieces of information to produce a final velocity map that is both accurate in its velocity values and sharp at geological boundaries.
            </li>

            <li>
                <strong>One AI to Rule Them All? (Foundation Models & Adaptation):</strong> Inspired by the success of large language models (LLMs like GPT) and large vision models, researchers are exploring <strong class="key-concept">FWI foundation models</strong>. These are massive neural networks pre-trained on enormous, diverse seismic datasets (like the combined OpenFWI). The goal is for these models to learn general principles of wave physics and geology. Then, for a specific new survey or geological setting, instead of training a model from scratch (which is costly), this pre-trained foundation model can be quickly adapted. Techniques like <strong class="key-concept">Parameter-Efficient Fine-Tuning (PEFT)</strong>, particularly <strong class="key-concept">Low-Rank Adaptation (LoRA)</strong>, allow this adaptation by training only a tiny fraction of the model's parameters. This is computationally efficient and surprisingly effective, especially for generalizing to data distributions not seen during pre-training.

                <div class="accordion-item">
                    <button class="accordion-button">What are Foundation Models, PEFT, and LoRA?</button>
                    <div class="accordion-content">
                        <p><strong>Foundation Models:</strong> Think of a huge, general-purpose AI model (like GPT for text, or a similar large model trained on vast amounts of seismic data). It's pre-trained on a massive, diverse dataset to gain a broad understanding of its domain (language, vision, or in our case, seismic waves and geology). It serves as a powerful starting point.</p>
                        <p><strong>Parameter-Efficient Fine-Tuning (PEFT):</strong> When you want to use a foundation model for a *specific* new task (like FWI for a particular region), you don't need to retrain the entire huge model. PEFT techniques allow you to adapt the model by training only a small number of *new* or *modified* parameters, while keeping most of the original model's knowledge frozen. This is much faster and requires less computing power than full retraining.</p>
                        <p><strong>Low-Rank Adaptation (LoRA):</strong> A popular PEFT technique. Instead of changing the foundation model's existing "brain cells" (weights), LoRA adds small, trainable "adapter modules" alongside them. These adapters learn the task-specific adjustments. Because these adapters are cleverly designed to be "low-rank" (mathematically simple), they add very few trainable parameters, making adaptation highly efficient.</p>

                        <h5>More Rigorous Definition</h5>
                        <p><strong>Foundation Models:</strong> Large-scale neural network models pre-trained on vast quantities of broad data (e.g., diverse seismic datasets like OpenFWI for FWI foundation models). They are designed to capture general-purpose representations and capabilities within their domain, serving as a base model that can be adapted (fine-tuned) for various downstream tasks with potentially less task-specific data and computation.</p>
                        <p><strong>Parameter-Efficient Fine-Tuning (PEFT):</strong> A collection of techniques aimed at adapting large pre-trained foundation models to downstream tasks by modifying only a small subset of the model's parameters, or by adding a small number of new parameters, while keeping the majority of the original parameters frozen. This significantly reduces the computational cost (memory, time) and storage requirements associated with fine-tuning large models.</p>
                        <p><strong>Low-Rank Adaptation (LoRA):</strong> A specific PEFT method that injects trainable low-rank matrices into the layers (e.g., linear, convolutional) of a pre-trained model during fine-tuning. For a weight matrix <span class="math-notation">Wâ‚€</span>, LoRA learns an update <span class="math-notation">Î”W = BA</span>, where <span class="math-notation">A</span> and <span class="math-notation">B</span> are low-rank matrices (<span class="math-notation">rank r â‰ª dimensions of Wâ‚€</span>). Only <span class="math-notation">A</span> and <span class="math-notation">B</span> are trained, drastically reducing the number of trainable parameters compared to full fine-tuning. The modified forward pass becomes <span class="math-notation">h = Wâ‚€x + Î±(BA)x</span>.</p>
                    </div>
                </div>
            </li>
        </ul>

        <p>
            Below, we delve into detailed reviews of a few representative papers from the 2022â€“2025 period that exemplify these exciting advancements, including simplified code examples to illustrate key concepts.
        </p>

        <h2 id="detailed-reviews">Detailed Paper Reviews (2022â€“2025)</h2>

        <h3 id="svit">
            <span class="podcast-link"><a href="https://notebooklm.google.com/notebook/5302dc6d-35a8-4598-acd5-241c3b01d03c/audio" target="_blank" rel="noopener">One-Fit-All Transformer Podcast</a></span>
            Seismic Velocity Inversion Transformer (SVIT, 2023)
        </h3>
        <p><strong>Reference:</strong> Yue Li and Baojun Yang, GEOPHYSICS, 88(4): R513â€“R533, 2023.</p>
        (*Note: The podcast link points to a general Transformer FWI concept, relevant here.*)

        <h4>Overview</h4>
        <p>
            SVIT stands out as one of a first applications of the powerful <strong class="key-concept">Transformer</strong> architecture, originally famed for its success in natural language processing (NLP), to the FWI problem. It directly addresses a limitation of prior CNN-based models (like U-Nets): CNNs rely on local convolutional filters, which excel at capturing nearby spatial patterns but may struggle to relate information across large distances in the seismic data record (e.g., connecting early arrivals at near sensors to late arrivals at far sensors).
        </p>
        <blockquote>
            <p>SVIT employs the Transformer's core mechanism: <strong class="key-concept">self-attention</strong>. Imagine the model possessing the ability to look at the entire seismic recording simultaneously and dynamically decide which parts are most relevant to understanding any other part, regardless of their separation in time or space.</p>
            <p>This "global receptive field" allows SVIT to build a more holistic understanding of the wave propagation patterns, potentially leading to better reconstruction of large-scale geological structures and continuous layers compared to locally-focused CNNs.</p>
        </blockquote>
         <div class="accordion-item">
            <button class="accordion-button">Refresher: Transformers & Self-Attention</button>
            <div class="accordion-content">
                <p>For definitions of Transformers and Self-Attention, please see the earlier accordion 
                <em>"What are CNNs, Transformers, and Self-Attention?"</em> in the <strong>Recent Advances</strong> section above.</p>
            </div>
        </div>


        <h4>Key Contributions</h4>
        <ul>
            <li>Pioneered the use of Transformer architecture for end-to-end FWI, showcasing the effectiveness of self-attention for modeling long-range dependencies in seismic waveforms.</li>
            <li>Reportedly achieved improved accuracy and geological consistency in velocity model reconstruction compared to baseline CNNs (e.g., U-Net) and even some traditional FWI approaches in specific test cases. The resulting velocity maps exhibited clearer structural features and interfaces.</li>
        </ul>

        <h4>Architecture</h4>
        <p>
            SVIT likely adopts an encoder-decoder structure, drawing inspiration from Vision Transformers (ViTs) used in computer vision. The key steps typically involve:
        </p>
        <ol>
            <li><strong>Input Patching:</strong> The input seismic data (often represented as a 2D array of time samples vs. receiver locations) is divided into smaller, non-overlapping patches, similar to breaking an image into tiles.</li>
            <li><strong>Linear Embedding:</strong> Each patch is flattened and linearly projected into a vector (a "token"). Positional embeddings are added to these tokens to retain information about the original location of each patch.</li>
            <li><strong>Transformer Encoder:</strong> These patch tokens, augmented with positional information, are processed through a stack of Transformer blocks. Each block contains a multi-head self-attention layer (allowing the model to attend to information from different perspectives) and a feed-forward neural network. This is where the model learns the complex, long-range relationships within the seismic data.</li>
            <li><strong>Decoder/Reconstruction Head:</strong> The processed tokens from the encoder are then fed into a decoder module, which reconstructs the 2D velocity map of the subsurface. This decoder might use transposed convolutions (sometimes called "deconvolutions") or other up-sampling techniques.</li>
        </ol>
        <p>
            Essentially, SVIT replaces the convolutional layers of a typical CNN encoder with Transformer blocks centered around the self-attention mechanism.
        </p>

        <h4>Training & Loss Function</h4>
        <p>
            SVIT is trained in a <strong class="key-concept">supervised learning</strong> setting. This means it learns from a large dataset of paired examples: input seismic data and the corresponding known "ground truth" velocity map. The dataset likely consists of synthetically generated examples covering various geological scenarios. During training, the model's predicted velocity map is compared to the ground truth map using a <strong class="key-concept">loss function</strong>. Common choices include pixel-wise losses like Mean Squared Error (MSE) or Mean Absolute Error (MAE). The goal of training is to adjust the model's internal parameters (weights) to minimize this loss. Performance is evaluated using metrics like MAE, MSE, and often the Structural Similarity Index (SSIM), which measures perceptual similarity between the predicted and true maps. SVIT was reported to show improvements in these metrics, indicating better reconstruction of velocity values, structure shapes, and interface locations.
        </p>

        <div class="accordion-item">
            <button class="accordion-button">What are Supervised Learning & Loss Functions?</button>
            <div class="accordion-content">
                <p><strong>Supervised Learning:</strong> Like teaching a student with an answer key. The AI model is shown input examples (seismic data) along with the correct outputs (velocity maps). It learns by comparing its predictions to the correct answers and adjusting itself to get closer.</p>
                <p><strong>Loss Function:</strong> A mathematical formula that measures how "wrong" the AI's prediction is compared to the correct answer. Common examples:</p>
                <ul>
                    <li><strong>Mean Squared Error (MSE):</strong> Calculates the average of the squared differences between predicted and true values. Penalizes large errors heavily.</li>
                    <li><strong>Mean Absolute Error (MAE):</strong> Calculates the average of the absolute differences. Less sensitive to outliers than MSE.</li>
                    <li><strong>Structural Similarity Index (SSIM):</strong> Measures the similarity between two images based on structure, contrast, and luminance. Often better captures visual quality than simple pixel differences.</li>
                </ul>
                <p>The goal of training is to minimize the value of the loss function.</p>

                <h5>More Rigorous Definition</h5>
                <p><strong>Supervised Learning:</strong> A machine learning paradigm where an algorithm learns a mapping function from input variables (<span class="math-notation">X</span>) to output variables (<span class="math-notation">Y</span>) based on a labeled dataset of input-output pairs (<span class="math-notation">{(xáµ¢, yáµ¢)}</span>). The goal is to approximate the function <span class="math-notation">f</span> such that <span class="math-notation">f(X)</span> predicts <span class="math-notation">Y</span> accurately on unseen data.</p>
                <p><strong>Loss Function (or Cost Function):</strong> A function <span class="math-notation">L(y<sub class="subscript">pred</sub>, y<sub class="subscript">true</sub>)</span> that quantifies the discrepancy between the model's prediction (<span class="math-notation">y<sub class="subscript">pred</sub></span>) and the true target value (<span class="math-notation">y<sub class="subscript">true</sub></span>). The objective of training is to find model parameters that minimize the average loss over the training dataset. Examples:</p>
                <ul>
                    <li><strong>MSE:</strong> <span class="math-notation">L = (1/N) Î£ (y<sub class="subscript">pred</sub> - y<sub class="subscript">true</sub>)Â²</span></li>
                    <li><strong>MAE:</strong> <span class="math-notation">L = (1/N) Î£ |y<sub class="subscript">pred</sub> - y<sub class="subscript">true</sub>|</span></li>
                    <li><strong>SSIM:</strong> A more complex function comparing local statistics (mean, variance, covariance) of image patches.</li>
                </ul>
            </div>
        </div>


        <h4>Reproducibility</h4>
        <p>
            The authors published their findings in the Geophysics journal but did not provide an official open-source code release. However, the architectural components (patch embedding, multi-head self-attention, MLP blocks) are standard building blocks within the Transformer ecosystem (particularly Vision Transformers). Researchers familiar with frameworks like PyTorch or TensorFlow could potentially reimplement the core ideas.
        </p>

        <h4>Conceptual Implementation (PyTorch): Simplified Transformer for FWI</h4>
        <p>Here's a conceptual PyTorch code snippet illustrating the core structure of a Transformer-based FWI model, inspired by SVIT.</p>
        <div class="code-container">
            <button class="toggle-code-btn">Toggle Code</button>
            <button class="copy-code-btn" title="Copy code">Copy</button>
            <pre class="line-numbers"><code class="language-python">
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """Convert seismic data (time x receivers image) into patch tokens."""
    def __init__(self, img_size=(1000, 70), patch_size=(16, 16), in_chans=1, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        # Use a Conv2d layer with stride equal to patch_size for efficient patch extraction & embedding
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (batch, in_chans, height, width) -> e.g., (B, 1, 1000, 70)
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # Project and flatten: (B, embed_dim, num_patches_h, num_patches_w) -> (B, embed_dim, num_patches_total)
        x = self.proj(x).flatten(2)
        # Transpose to get (B, num_patches_total, embed_dim) - standard Transformer input format
        x = x.transpose(1, 2)
        return x

class TransformerEncoderLayer(nn.Module):
    """Standard Transformer Encoder Block."""
    def __init__(self, dim, num_heads, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        # Multi-Head Self-Attention
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        # Feed-Forward Network (MLP)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(), # Common activation in Transformers
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        # Note: Typically includes stochastic depth (DropPath), omitted here for simplicity

    def forward(self, x):
        # Self-attention part + residual connection
        residual = x
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm) # Q, K, V are the same
        x = residual + attn_output # Additive residual connection

        # Feed-forward part + residual connection
        residual = x
        x_norm = self.norm2(x)
        mlp_output = self.mlp(x_norm)
        x = residual + mlp_output # Additive residual connection
        return x

class FWI_TransformerDecoder(nn.Module):
    """ A simple decoder to reconstruct the velocity map from tokens """
    def __init__(self, num_patches, embed_dim, output_size=(100, 100)):
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.output_size = output_size
        # Example: Use a linear layer to project back to the flattened image size
        self.head = nn.Linear(embed_dim, output_size[0] * output_size[1])
        # More complex decoders might use convolutional layers (ConvTranspose2d)

    def forward(self, x):
        # x: (B, num_patches, embed_dim)
        # Option 1: Use the representation of the first token (if using a [CLS] token like BERT)
        # cls_token_representation = x[:, 0]
        # Option 2: Average pool all patch tokens
        avg_pool_representation = x.mean(dim=1) # (B, embed_dim)
        # Project to the flattened output image size
        flat_output = self.head(avg_pool_representation) # (B, output_H * output_W)
        # Reshape to the desired 2D velocity map (ensure channels dimension is included)
        output_map = flat_output.view(-1, 1, self.output_size[0], self.output_size[1])
        return output_map

class SVIT_Inspired_Model(nn.Module):
    """Simplified Transformer-based FWI model."""
    def __init__(self, input_shape=(1000, 70), patch_size=(10, 10), embed_dim=128, depth=6, num_heads=8, output_shape=(100, 100), dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=input_shape, patch_size=patch_size, in_chans=1, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # Learnable positional embeddings provide spatial context for patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # Stack of Transformer Encoder layers
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderLayer(dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim) # Final normalization after encoder blocks

        # Decoder head to reconstruct the velocity map
        self.decoder = FWI_TransformerDecoder(num_patches=num_patches, embed_dim=embed_dim, output_size=output_shape)

        # Initialize weights (important for stable training)
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: (B, 1, time_steps, num_receivers) - e.g., (B, 1, 1000, 70)
        B = x.shape[0]

        # 1. Embed patches & Flatten
        tokens = self.patch_embed(x) # (B, num_patches, embed_dim)

        # 2. Add positional embedding
        tokens = tokens + self.pos_embed # Broadcasting applies pos_embed to all batches
        tokens = self.pos_drop(tokens)

        # 3. Pass through Transformer encoder blocks
        for blk in self.encoder_blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens) # Apply final normalization

        # 4. Decode to velocity map
        velocity_map = self.decoder(tokens) # (B, 1, output_H, output_W)

        return velocity_map

# --- Example Usage ---
# Define model parameters matching example data
model = SVIT_Inspired_Model(input_shape=(1000, 70), patch_size=(10, 10), embed_dim=128, depth=4, num_heads=8, output_shape=(100, 100))

# Create a dummy batch of seismic data
# Batch size = 4, Channels = 1, Height = 1000 (time steps), Width = 70 (receivers)
seismic_batch = torch.randn(4, 1, 1000, 70)

# Perform a forward pass
velocity_output = model(seismic_batch)

# Print the shape of the output velocity map
print(f"Output velocity map shape: {velocity_output.shape}") # Expected: torch.Size([4, 1, 100, 100])
            </code></pre>
        </div>
        <p>
            This sketch highlights the key stages: patch embedding, adding positional information, processing through Transformer encoder blocks using self-attention, and finally decoding the processed information back into a 2D velocity map. Training would involve feeding batches of seismic data, getting the predicted velocity map, comparing it to the known true map using a loss function (like MSE), and updating the model's weights via backpropagation.
        </p>

        <h3 id="vifnet">Velocity Interface Fusion Network (VIF-Net, 2024)</h3>
        <p><strong>Reference:</strong> Z. Wang et al., Computers & Geosciences, 174: 105834, 2024.</p>

        <h4>Overview</h4>
        <p>
            A persistent challenge in FWI, whether traditional or DL-based, is accurately resolving sharp boundaries between different geological units (e.g., layers of rock with distinct properties) or across faults. Standard DL models, particularly those based on smooth loss functions like MSE, often produce velocity maps that appear blurry or smoothed-out across these crucial <strong class="key-concept">interfaces</strong>. VIF-Net tackles this specific issue with an innovative two-stage, multi-network approach.
        </p>
        <div class="accordion-item">
            <button class="accordion-button">What are Interfaces in Geology?</button>
            <div class="accordion-content">
                 <p>Think of the boundaries between different layers in a cake or different types of rock underground. These are interfaces. They represent locations where the properties of the material change abruptly (e.g., from sandstone to shale, or across a fault where rocks have shifted). Accurately mapping these interfaces is critical for understanding the geological structure.</p>
                 <h5>More Rigorous Definition</h5>
                 <p>In geophysics and geology, an interface refers to a surface or boundary separating two media with different physical properties (e.g., seismic velocity, density, electrical resistivity). Examples include stratigraphic boundaries between rock layers, unconformities, fault planes, or the boundary of a fluid reservoir. Seismic waves reflect and refract at these interfaces, and accurately imaging their location and character is a primary goal of seismic exploration and FWI.</p>
            </div>
        </div>

        <blockquote>
            <p>Think of creating a detailed geological map. You need two key pieces of information: the elevation (representing seismic velocity) at every point, and the precise location of sharp changes like cliffs or riverbanks (representing geological interfaces).</p>
            <p>VIF-Net essentially trains specialist AIs for each task: one estimates the overall velocity landscape, another pinpoints the locations of sharp boundaries. Then, a final "fusion" AI combines this information to create a map that respects both the smooth velocity variations and the sharp interface locations.</p>
        </blockquote>

        <h4>Key Contributions</h4>
        <ul>
            <li>Introduces a structured two-stage workflow tailored for interface sharpness:
                <ol>
                    <li><strong>Pre-inversion Stage:</strong> Two parallel neural networks analyze the input seismic data simultaneously.
                        <ul>
                           <li>Network 1 (<code>VIF-V</code>): Predicts a preliminary, potentially smooth, velocity map.</li>
                           <li>Network 2 (<code>VIF-I</code>): Predicts an "interface map," highlighting the probability or location of sharp geological boundaries (essentially performing an edge or boundary detection task).</li>
                        </ul>
                    </li>
                    <li><strong>Fusion Stage:</strong> A third network takes the outputs from Stage 1 (the preliminary velocity map and the interface location map) as combined input. It intelligently merges this information to produce a final, refined velocity map characterized by significantly sharper and more accurate interfaces.</li>
                </ol>
            </li>
            <li>Demonstrates significantly improved recovery of sharp geological features like layer boundaries and faults compared to conventional single-network FWI models that tend to oversmooth these critical details.</li>
        </ul>

        <h4>Architecture</h4>
        <p>
            The paper suggests that the networks in Stage 1 are likely CNN-based. <code>VIF-V</code> could be a standard regression network, possibly a U-Net architecture, trained to output velocity values. <code>VIF-I</code> would be structured as an image segmentation network (architectures like U-Net or DeepLabv3+ are common choices), trained to output a binary or probability map indicating interface locations. The Stage 2 "Fusion Network" could be another U-Net-like architecture. Its crucial role is to take the intermediate velocity map and the interface map as multi-channel input and learn how to refine the velocity map, paying particular attention to preserving or sharpening features indicated by the interface map.
        </p>
        <div class="accordion-item">
            <button class="accordion-button">What is Image Segmentation and related Losses?</button>
            <div class="accordion-content">
                <p><strong>Image Segmentation:</strong> Imagine coloring in a map where each region (like a country or a lake) gets a specific color. Image segmentation is similar: an AI model looks at an image and assigns a label (like "rock layer A," "rock layer B," or "fault boundary") to every single pixel. VIF-Net's interface network (<code>VIF-I</code>) performs segmentation to identify pixels belonging to sharp boundaries.</p>
                <p><strong>Segmentation Losses (BCE, Dice):</strong> These are ways to measure how well the AI's segmentation map matches the true map. </p>
                <ul>
                    <li><strong>Binary Cross-Entropy (BCE):</strong> Commonly used when there are only two classes (e.g., "interface" vs. "not interface"). It measures the difference between the predicted probability and the true label (0 or 1) for each pixel.</li>
                    <li><strong>Dice Coefficient (or Dice Loss):</strong> Measures the overlap between the predicted segmentation and the true segmentation. A score of 1 means perfect overlap, 0 means no overlap. It's often good for handling situations where the target regions (like interfaces) are small compared to the background.</li>
                </ul>

                <h5>More Rigorous Definition</h5>
                <p><strong>Image Segmentation:</strong> The task of partitioning a digital image into multiple segments (sets of pixels, also known as superpixels). The goal is typically to assign a class label to each pixel in the image, such that pixels with the same label share certain characteristics (e.g., semantic meaning, visual properties). In VIF-Net, the interface network performs semantic segmentation to classify pixels as belonging to a geological interface or not.</p>
                <p><strong>Segmentation Losses:</strong></p>
                 <ul>
                    <li><strong>Binary Cross-Entropy (BCE) Loss:</strong> For binary segmentation (two classes), measures the negative log-likelihood of the predicted probabilities assuming a Bernoulli distribution for each pixel. <span class="math-notation">L<sub class="subscript">BCE</sub> = - (1/N) Î£ [ y log(p) + (1-y) log(1-p) ]</span>, where <span class="math-notation">y</span> is the true label (0 or 1) and <span class="math-notation">p</span> is the predicted probability for class 1.</li>
                    <li><strong>Dice Coefficient / Loss:</strong> Measures spatial overlap. The Dice coefficient is <span class="math-notation">DSC = 2 * |X âˆ© Y| / (|X| + |Y|)</span>, where X is the predicted set of pixels and Y is the true set. Dice Loss is often defined as <span class="math-notation">L<sub class="subscript">Dice</sub> = 1 - DSC</span>. It is particularly useful for unbalanced datasets where the target region is small.</li>
                </ul>
            </div>
        </div>


        <h4>Training Setup</h4>
        <p>
            Training VIF-Net effectively requires ground truth data for both the velocity map and the interface locations. The interface ground truth can often be derived from the true velocity map by applying edge detection algorithms (e.g., finding regions with high gradients). The networks might be trained end-to-end, or potentially sequentially. <code>VIF-V</code> would typically use a regression loss like MSE or MAE on the velocity values. <code>VIF-I</code> would use a segmentation loss, such as Binary Cross-Entropy (BCE) or the Dice coefficient, comparing the predicted interface map to the true interface locations. The Fusion Network is trained to minimize the error (e.g., MSE) in the final velocity output, possibly with additional loss terms that encourage sharpness guided by the interface input. The authors likely tested VIF-Net on synthetic datasets featuring distinct interfaces, such as those found in the "Fault" or "Style" categories of the OpenFWI benchmark.
        </p>

        <h4>Results</h4>
        <p>
            VIF-Net showcased visually clearer and quantitatively more accurate delineation of faults and layer boundaries in test cases compared to baseline methods. This enhanced ability to resolve sharp discontinuities is vital for accurate geological interpretation, such as identifying reservoir seals or mapping fracture networks.
        </p>

        <h4>Open Source Availability</h4>
        <p>
            At the time of this review, an official code repository from the authors does not appear to be publicly available. However, the conceptual framework relies on combining well-established CNN architectures for regression and segmentation tasks, making reimplementation feasible for researchers skilled in deep learning.
        </p>


        <h3 id="ifwi">
             <span class="podcast-link"><a href="https://notebooklm.google.com/notebook/d3bd4da6-8b26-4bb9-aa2f-5de75c5ba916/audio" target="_blank" rel="noopener">InversionNet3D Podcast</a></span>
            Implicit Full Waveform Inversion (IFWI, 2022/2023)
        </h3>
        <p><strong>Reference:</strong> Jian Sun and Kristopher Innanen, arXiv:2209.03525 (2022), J. Geophys. Res.: Solid Earth (2023).</p>
         (*Note: Podcast link relates to advanced DL architectures for FWI, conceptually relevant to complex methods like IFWI.*)


        <h4>Overview</h4>
        <p>
            Implicit Full Waveform Inversion (IFWI) represents a paradigm shift from typical supervised DL approaches. Instead of training a network to map input seismic data directly to an output velocity grid (like image-to-image translation), IFWI uses a neural network to <span class="key-concept">implicitly represent</span> the velocity field itself as a continuous function.
        </p>
        <div class="accordion-item">
            <button class="accordion-button">What is an Implicit Neural Representation?</button>
            <div class="accordion-content">
                <p>Instead of storing information like a picture (a grid of pixels), imagine having a magic formula (the neural network). You give the formula coordinates (like latitude and longitude, or x and z depth), and it calculates the value at that exact point (like the temperature, or in FWI, the seismic velocity). The network *is* the representation; its internal parameters (weights) define the continuous field.</p>

                <h5>More Rigorous Definition</h5>
                <p>An Implicit Neural Representation (INR), also known as a coordinate-based network, uses a neural network (often an MLP) to parameterize a signal or field as a continuous function of its coordinates. Instead of discretizing the signal onto a grid, the INR <span class="math-notation">f<sub class="subscript">Î¸</sub>(<span class="math-notation">x</span>)</span> maps input coordinates <span class="math-notation">x</span> (e.g., spatial coordinates <span class="math-notation">(x, y, z)</span>) to the signal value at that location (e.g., velocity <span class="math-notation">v</span>), where <span class="math-notation">Î¸</span> represents the network's parameters. The network weights <span class="math-notation">Î¸</span> are optimized to fit observed data or satisfy certain constraints (like PDEs in PINNs).</p>
            </div>
        </div>

        <blockquote>
            <p>Imagine the velocity map isn't stored as a grid of discrete pixel values. Instead, you have a small, flexible mathematical formula (parameterized by a neural network) that can tell you the precise velocity at <span style="font-style: normal; font-weight: bold;">any</span> continuous coordinate (x, z) you query.</p>
            <p>This network, often a simple <strong class="key-concept">Multi-Layer Perceptron (MLP)</strong>, initially knows nothing (random weights). It's then optimized directly against the observed seismic data. How?
            <br>1. Use the MLP to generate a velocity map on a grid (by querying it at many points).
            <br>2. Simulate seismic waves through this map using a physics engine.
            <br>3. Compare the simulated echoes to the real, observed echoes.
            <br>4. Calculate how to adjust the MLP's internal parameters (weights) to make the simulated echoes better match the real ones.
            <br>The network itself *becomes* the inversion result; its weights encode the final velocity model.</p>
        </blockquote>
        <p>
            This falls under the umbrella of <strong class="key-concept">Physics-Informed Neural Networks (PINNs)</strong> or coordinate-based representations. The crucial aspect is that the network learns *only* from the mismatch between simulated and observed data, guided by the physics of wave propagation embedded in the simulator. It requires no pre-existing library of velocity map examples for training.
        </p>

        <div class="accordion-item">
            <button class="accordion-button">What is a Multi-Layer Perceptron (MLP)?</button>
            <div class="accordion-content">
                <p>An MLP is one of the simplest types of "deep learning" networks. It consists of layers of artificial neurons. Information flows from an input layer, through one or more "hidden" layers, to an output layer. Each connection between neurons has a weight, and the network learns by adjusting these weights. Think of it as a series of adjustable knobs and switches that transform the input data into the desired output.</p>

                <h5>More Rigorous Definition</h5>
                <p>A Multi-Layer Perceptron (MLP) is a class of feedforward artificial neural network (ANN). It consists of at least three layers of nodes: an input layer, one or more hidden layers, and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation function. MLPs utilize supervised learning with backpropagation for training. They are universal function approximators, meaning they can theoretically approximate any continuous function given enough hidden units and appropriate weights.</p>
            </div>
        </div>

        <h4>Key Contributions</h4>
        <ul>
            <li>Demonstrates that representing the velocity field implicitly with a neural network can effectively mitigate the <strong class="key-concept">cycle-skipping</strong> problem that plagues traditional FWI. The inherent smoothness and flexibility of the neural network representation seem to help the optimization process avoid getting trapped in incorrect local minima, even when starting from a poor initial guess (e.g., a constant velocity).</li>
            <li>Offers a natural framework for uncertainty quantification. By incorporating techniques like Monte Carlo dropout during the optimization phase, one can obtain a distribution of slightly different velocity models consistent with the data. Analyzing the variance across these models can highlight which parts of the subsurface are well-constrained by the seismic data and which parts remain uncertain.</li>
            <li>Provides a flexible framework that could potentially incorporate various physical constraints or fuse different types of geophysical data more easily than grid-based methods.</li>
        </ul>

        <h4>Architecture</h4>
        <p>
            The core component is usually a simple Multi-Layer Perceptron (MLP). This network takes 2D (or 3D) spatial coordinates (x, z) as input and outputs a single scalar value representing the seismic velocity at that location. To enable the network to represent fine details and high-frequency variations in the velocity field, techniques like <strong class="key-concept">positional encoding</strong> (similar to those used in Transformers and NERFs) or using periodic activation functions (like sine, as in <strong class="key-concept">SIRENs</strong> - Sinusoidal Representation Networks) are often employed. This velocity-representing network is tightly coupled with a <strong class="key-concept">differentiable wave simulation engine</strong>. "Differentiable" means that we can automatically compute gradients (how the output changes with respect to input) through the entire simulation process, which is essential for optimizing the neural network's weights using standard backpropagation techniques.
        </p>

        <div class="accordion-item">
            <button class="accordion-button">What are Positional Encoding, SIRENs, and Differentiable Simulators?</button>
            <div class="accordion-content">
                <h4>Positional Encoding (Conceptual Introduction)</h4>
                <p>Positional encoding is a method used to help neural networks, especially coordinate-based ones, learn patterns that can change quickly with respect to position. Standard MLPs, which often rely on activation functions like ReLU, can struggle to capture high-frequency variationsâ€”think of abrupt changes in a physical field or sharp edges in an image. By transforming simple coordinates (like x or z) into richer representations, the network can more readily model small-scale or quickly changing features. This helps in tasks such as reconstructing detailed velocity profiles or capturing subtle variations in a signal.</p>

                <h4>Positional Encoding (Rigorous Details)</h4>
                <p>Concretely, positional encoding takes a low-dimensional coordinate vector
                <span class="math-notation">x</span> and maps it into a higher-dimensional space using periodic functions. A typical approach is to produce terms of the form
                <span class="math-notation">sin(2<sup class="superscript">k</sup>Ï€x)</span> and
                <span class="math-notation">cos(2<sup class="superscript">k</sup>Ï€x)</span> for various integer frequencies
                <span class="math-notation">k</span>. This set of sine and cosine functions acts like a basis that spans both low- and high-frequency components. When passed into an MLP, these encoded features counteract the networkâ€™s tendencyâ€”known as spectral biasâ€”to focus on smoother, lower-frequency patterns, thereby enabling it to reconstruct signals with sharp gradients and detailed oscillations.</p>

                <h4>SIRENs (Conceptual Introduction)</h4>
                <p>SIRENs, or Sinusoidal Representation Networks, turn the usual practice of using piecewise-linear or smooth nonlinearities on its head by making the sine function their core activation. Where networks with ReLU activations might need many layers to approximate highly oscillatory signals, SIRENs capture such signals more naturally because the sine function itself is oscillatory. They excel at learning tasks like representing audio, images, or physical fields where fine-grained details and derivatives are essential.</p>

                <h4>SIRENs (Rigorous Details)</h4>
                <p>In SIRENs, each layer computes outputs of the form <span class="math-notation">sin(Ï‰â‚€Wx + b)</span>, guided by a specific initialization scheme to ensure stable training. The sine activationâ€™s periodic nature directly encodes oscillations into each layer, granting the network a built-in capacity for modeling complex patterns. This design also aids in computing derivatives of the learned representation, proving useful for physics-driven tasks that involve partial differential equations. As a result, SIRENs offer high-fidelity reconstruction of continuous signals and fields, often outperforming standard MLP-based approaches for problems requiring detailed, precise representations.</p>

                <h4>Differentiable Simulator (Conceptual Introduction)</h4>
                <p>A computer program that simulates a physical process (like wave propagation) but is written in a way that allows automatic calculation of derivatives (gradients). Think of it as a simulator that not only tells you the result (simulated echoes) but also exactly how that result would change if you slightly tweaked any input parameter (like the velocity at one point in the model). This is crucial for training physics-informed models using gradient-based optimization.</p>

                <h4>Differentiable Simulator/Solver (Rigorous Details)</h4>
                <p>A numerical solver for a physical system (e.g., a PDE solver for the wave equation) implemented within a framework that supports automatic differentiation (AD), such as PyTorch or TensorFlow. This allows gradients of the simulation output (e.g., synthetic seismograms) with respect to the input parameters (e.g., velocity model parameters, network weights) to be computed efficiently and accurately via backpropagation, enabling end-to-end gradient-based optimization of models that incorporate the simulator.</p>
            </div>
        </div>

        <h4>Optimization (Inversion) Procedure</h4>
        <p>
            Unlike supervised DL, IFWI doesn't have a separate "training" phase on a large dataset. Each inversion performed on a specific set of observed seismic data is its own optimization problem:
        </p>
        <ol>
            <li>Initialize the weights of the velocity-representing neural network (e.g., MLP) randomly.</li>
            <li>Define a grid of coordinates covering the desired subsurface model area.</li>
            <li>Generate the current velocity map by querying the neural network at all grid coordinates.</li>
            <li>Input this velocity map into the differentiable wave simulator, along with source/receiver locations, to produce simulated seismic data (echoes).</li>
            <li>Calculate a loss function comparing the simulated seismic data with the actual observed seismic data (e.g., L2 norm of the difference).</li>
            <li>Use automatic differentiation (provided by frameworks like PyTorch or TensorFlow) to compute the gradient of the loss with respect to the neural network's weights. This gradient indicates how to change the weights to reduce the data mismatch.</li>
            <li>Update the network's weights using an optimization algorithm (e.g., Adam, L-BFGS).</li>
            <li>Repeat steps 3-7 iteratively until the loss converges (i.e., the simulated data closely matches the observed data).</li>
        </ol>
        <p>
            The final, optimized neural network <span style="font-style: normal; font-weight: bold;">is</span> the result. It implicitly holds the converged velocity model.
        </p>

        <h4>Evaluation</h4>
        <p>
            IFWI was typically evaluated on complex synthetic benchmark models (like Marmousi or BP). The key success metric is its ability to recover the true velocity structure with high fidelity, especially starting from very simple or inaccurate initial models where traditional FWI methods often fail due to cycle-skipping. Resolution and accuracy are compared against ground truth and traditional FWI results.
        </p>

        <h4>Reproducibility</h4>
        <p>
            Implementing IFWI requires integrating a neural network framework (like PyTorch) with a differentiable wave simulation library or code (e.g., DeepWave, Salvus Pytorch, or custom implementations using finite differences and autograd). While the specific code from the original authors may not be public, the core concepts draw from the active research areas of PINNs and neural implicit representations, where various tools and examples are becoming increasingly available within the scientific computing community.
        </p>

        <h4>Conceptual Implementation (PyTorch): Coordinate-based Velocity Network</h4>
        <p>This snippet focuses on defining the velocity network and outlining the optimization loop. A functional differentiable wave simulator is assumed to exist.</p>
        <div class="code-container">
            <button class="toggle-code-btn">Toggle Code</button>
             <button class="copy-code-btn" title="Copy code">Copy</button>
            <pre class="line-numbers"><code class="language-python">
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F # For loss calculation

# --- Assumed Components (Replace with actual implementations) ---
# 1. obs_data: Tensor containing the real observed seismic data.
#    Shape: e.g., [num_shots, num_receivers, num_timesteps]
# 2. simulate_wave_autodiff(velocity_map, source_coords, receiver_coords, wave_params):
#    A *differentiable* function. Takes a velocity map tensor, source/receiver info,
#    and simulation parameters. Returns simulated seismic data tensor.
#    Crucially, gradients must flow from output data back to input velocity_map.
# 3. coords_tensor: Tensor of (x, z) coordinates covering the model space.
#    Shape: [nx * nz, 2], normalized (e.g., to [-1, 1] or [0, 1]).
# 4. nx, nz: Dimensions of the desired velocity grid output.
# 5. source_coords, receiver_coords, wave_params: Simulation setup details.
# 6. obs_data size for demonstration (replace with real data).

class VelocityFieldNN(nn.Module):
    """MLP mapping (x, z) coordinates to velocity scalar."""
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=1, n_layers=6, activation=nn.ReLU()):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), activation]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), activation])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, coords):
        # coords: Tensor shape (N, 2)
        velocity_values = self.network(coords) # Shape (N, 1)
        return velocity_values

# --- Setup for a Hypothetical Inversion ---
nx, nz = 128, 128 # Example grid size
# Generate normalized coordinates
x_lin = torch.linspace(-1, 1, nx)
z_lin = torch.linspace(-1, 1, nz)
grid_x, grid_z = torch.meshgrid(x_lin, z_lin, indexing='ij')
coords_tensor = torch.stack([grid_x.flatten(), grid_z.flatten()], dim=-1).float().requires_grad_(False)

dummy_obs_data_shape = [1, 100, 1000] # Example: shots, receivers, time
obs_data = torch.randn(dummy_obs_data_shape) # Replace with loading real data

vel_nn = VelocityFieldNN(input_dim=2, hidden_dim=128, n_layers=4, activation=nn.Tanh()) # Tanh can work well
optimizer = optim.Adam(vel_nn.parameters(), lr=5e-4)
num_iterations = 1000

print("Starting IFWI optimization loop...")
for it in range(num_iterations):
    optimizer.zero_grad() 

    # 1. Query the network at all grid coordinates -> velocity map
    velocity_values_flat = vel_nn(coords_tensor) # (nx*nz, 1)
    velocity_map = velocity_values_flat.view(nx, nz)

    # 2. Differentiable wave simulation placeholder
    sim_data = torch.rand_like(obs_data) * velocity_map.mean()

    # 3. Calculate the loss (data mismatch)
    loss = F.mse_loss(sim_data, obs_data)

    # 4. Backprop
    loss.backward()
    optimizer.step()

    if it % 50 == 0 or it == num_iterations - 1:
        print(f"Iteration {it}/{num_iterations}, Loss: {loss.item():.8f}")

print("Optimization finished.")
            </code></pre>
        </div>
        <p>This conceptual code illustrates the optimization cycle central to IFWI. The critical dependency is the differentiable wave simulator that links the neural network's output (the velocity field) to the observable data, allowing gradient-based optimization of the network's parameters to minimize the data mismatch.</p>


        <h3 id="fwigan">
             <span class="podcast-link"><a href="https://notebooklm.google.com/notebook/63e5ca93-6fdb-4d3a-8fb4-4beb16bb65a1/audio" target="_blank" rel="noopener">FWIGAN Podcast</a></span>
             FWI-GAN: Physics-Informed Generative Adversarial Network (2023)
        </h3>
        <p><strong>Reference:</strong> Fangshu Yang and Jianwei Ma, J. Geophys. Res.: Solid Earth, 128(4), e2022JB025947, 2023.</p>

        <h4>Overview</h4>
        <p>
            FWI-GAN introduces another sophisticated AI technique, <strong class="key-concept">Generative Adversarial Networks (GANs)</strong>, into the physics-informed FWI landscape. It cleverly frames the inversion problem as a competitive game between two neural networks, guided by wave physics, to learn the subsurface structure directly from observed data without needing paired examples.
        </p>

        <div class="accordion-item">
            <button class="accordion-button">Refresher: GANs</button>
            <div class="accordion-content">
                <p>For an introduction to Generative Adversarial Networks, see the earlier accordion
                <em>"What are Deep Learning, U-Nets, and GANs in this context?"</em> in the Introduction section.</p>
            </div>
        </div>

        <blockquote>
            <p>Imagine a game between an "artist" (the <strong class="key-concept">Generator</strong>) and an "art critic" (the <strong class="key-concept">Discriminator</strong> or Critic).</p>
            <ol>
                <li>The Generator attempts to create a realistic velocity map.</li>
                <li>A physics simulator predicts the seismic echoes (synthetic data) that would result from the Generator's map.</li>
                <li>The Critic is shown two sets of seismic data: the *real* observed echoes from the field survey, and the *synthetic* echoes generated from the Generator's map.</li>
                <li>The Critic's job is to learn to distinguish between the real and synthetic seismic data ("real" vs. "fake").</li>
                <li>The Generator's goal is to create velocity maps whose synthetic echoes are so realistic that they successfully fool the Critic into thinking they are real.</li>
            </ol>
            <p>Through this adversarial process, the Generator is pushed to produce velocity maps that accurately explain the observed seismic data, as judged by the physics simulator and the discerning Critic.</p>
        </blockquote>
        <p>
            A significant advantage of this approach is its <strong class="key-concept">unsupervised</strong> nature regarding velocity maps. It does not require a training dataset of known (seismic data, true velocity map) pairs. Instead, it learns directly from the single set of observed seismic data for the specific area being investigated.
        </p>

        <div class="accordion-item">
            <button class="accordion-button">What is Unsupervised Learning?</button>
            <div class="accordion-content">
                 <p>Unlike supervised learning (with an answer key), unsupervised learning is like giving the AI a pile of unsorted data and asking it to find patterns or structure on its own, without any predefined correct answers. Examples include clustering data points into groups or learning to generate new data similar to the input data (like GANs).</p>
                 <h5>More Rigorous Definition</h5>
                 <p>Unsupervised learning is a machine learning paradigm where the algorithm learns patterns from unlabeled data (<span class="math-notation">X</span>) without corresponding output variables (<span class="math-notation">Y</span>). The goal is typically to model the underlying structure or distribution in the data, for tasks such as dimensionality reduction, clustering, density estimation, or generative modeling. FWI-GAN is unsupervised with respect to the velocity models, as it doesn't need ground truth <span class="math-notation">v</span> maps for training; it learns by comparing distributions of observed and simulated *seismic data*. </p>
            </div>
        </div>

        <h4>Key Contributions</h4>
        <ul>
            <li>Presents an unsupervised FWI method that embeds a physics simulator within a GAN framework, removing the need for labeled ground truth velocity maps during training/inversion.</li>
            <li>Demonstrates potential robustness against cycle-skipping and perhaps better handling of noise compared to methods relying solely on pixel-wise loss functions. GANs often optimize based on the overall distribution or "realism" of the data, which can be less sensitive to minor discrepancies. Variations like <strong class="key-concept">Wasserstein GAN (WGAN)</strong> with gradient penalty (GP) are often used for stability.</li>
            <li>Reportedly achieved high-quality inversion results on challenging benchmark models (e.g., Marmousi), comparing favorably against traditional FWI and some supervised DL methods.</li>
        </ul>

        <div class="accordion-item">
            <button class="accordion-button">What is Wasserstein GAN (WGAN) and Gradient Penalty (GP)?</button>
            <div class="accordion-content">
                <p>Standard GANs can be tricky to train â€“ sometimes the Generator and Discriminator stop learning effectively. WGAN is an improved version that uses a different mathematical way (the "Wasserstein" or "Earth Mover's" distance) to measure the difference between real and fake data distributions. This often makes training smoother and more stable.</p>
                <p>The Gradient Penalty (GP) is an additional technique used with WGAN. It helps ensure the Discriminator behaves nicely (doesn't change too abruptly), which further stabilizes the training process and prevents common GAN failure modes.</p>

                <h5>More Rigorous Definition</h5>
                <p><strong>Wasserstein GAN (WGAN):</strong> A modification of the standard GAN framework that minimizes an approximation of the Wasserstein-1 distance (or Earth Mover's distance) between the real data distribution (<span class="math-notation">P<sub class="subscript">r</sub></span>) and the generated data distribution (<span class="math-notation">P<sub class="subscript">g</sub></span>). This distance metric provides smoother gradients than the Jensen-Shannon divergence used implicitly in standard GANs, leading to more stable training and better correlation between the loss value and sample quality. WGANs typically require the discriminator (called the "critic" in WGAN terminology) to be 1-Lipschitz continuous.</p>
                <p><strong>Gradient Penalty (GP):</strong> A technique introduced in "Improved Training of Wasserstein GANs" (Gulrajani et al., 2017) to enforce the Lipschitz constraint on the critic in WGANs more effectively than the original weight clipping method. It adds a penalty term to the critic's loss function that encourages the gradient norm of the critic's output with respect to its input to be close to 1 for points sampled along straight lines between real and generated data points.</p>
            </div>
        </div>

        <h4>Methodology</h4>
        <p>
            The FWI-GAN system involves these core components:
        </p>
        <ol>
            <li><strong>Generator Network (G):</strong> Typically a neural network (e.g., a CNN like U-Net) that takes a random noise vector (or potentially a low-resolution initial model) as input and outputs a candidate velocity map.</li>
            <li><strong>Differentiable Physics Simulator (F):</strong> Same role as in IFWI; takes the velocity map from G and simulates the corresponding seismic data (<code>sim_data = F(G(z))</code>, where <code>z</code> is input noise).</li>
            <li><strong>Discriminator Network (D) / Critic:</strong> A neural network (often a CNN classifier) trained to distinguish between real and simulated data. It takes seismic data (either real <code>obs_data</code> or simulated <code>sim_data</code>) as input and outputs a score indicating how "real" it believes the data is (or a score related to the Wasserstein distance in WGAN).</li>
            <li><strong>Adversarial Training Loop:</strong> The Generator and Discriminator are trained iteratively:
                <ul>
                    <li><strong>Train Discriminator/Critic:</strong> Update D's weights to maximize its ability to correctly classify real data as real and simulated data as fake (or maximize the Wasserstein distance estimate).</li>
                    <li><strong>Train Generator:</strong> Update G's weights to minimize the Discriminator's output score for the simulated data (i.e., make <code>D(F(G(z)))</code> indicate "real", or minimize the Wasserstein distance estimate from G's perspective). This means G learns to generate velocity maps that produce highly realistic seismic data, effectively "fooling" D.</li>
                </ul>
            </li>
        </ol>
        <p>
            This alternating optimization process drives the Generator to produce velocity models that accurately explain the observed seismic data according to the embedded wave physics. Variations like Wasserstein GAN (WGAN) with gradient penalty are often used for more stable training.
        </p>

        <h4>Results</h4>
        <p>
            FWI-GAN demonstrated successful recovery of complex geological structures in standard benchmarks like the Marmousi model. It showed promise in handling strong velocity contrasts and reducing artifacts often seen in traditional methods, particularly when the starting model is poor. Its unsupervised nature makes it particularly attractive for real-world applications where accurate ground truth velocity models are rarely available.
        </p>

        <h4>Open Source Availability</h4>
        <p>
            Yes! A significant advantage for reproducibility and further research is that the authors <strong style="color: green;">released their PyTorch code implementation</strong>. It utilizes the DeepWave library for the differentiable wave simulation. This allows other researchers to build upon, test, and adapt their method. (See reference section for link).
        </p>


        <h3 id="foundation">
             <span class="podcast-link"><a href="https://notebooklm.google.com/notebook/f4f53cb4-42dc-4b7f-95c7-478d750d20a7/audio" target="_blank" rel="noopener">Big Data Impact Podcast</a></span>
            Foundation Models and Parameter-Efficient Fine-Tuning (PEFT) with LoRA (2024)
        </h3>
        <p><strong>Reference:</strong> D. Maiti et al., arXiv:2402.19510, 2024.</p>
         (*Note: Podcast link relates to the importance of large datasets, key for training foundation models.*)

        <h4>Overview</h4>
        <p>
            This recent work imports a powerful concept from the broader AI worldâ€”<strong class="key-concept">foundation models</strong>â€”into the domain of FWI. The core idea is analogous to Large Language Models (LLMs) like GPT: pre-train a single, very large, and capable neural network on a massive and highly diverse dataset. For FWI, this means training on datasets like the comprehensive OpenFWI benchmark, encompassing many different geological styles. The goal is for this foundation model to learn generalizable representations of seismic wave phenomena and subsurface structures. Then, rather than training specialized models from scratch for each new seismic survey or geological target (which is computationally expensive and requires significant data), this powerful pre-trained model can be rapidly and efficiently adapted to the specific task at hand.
        </p>

        <div class="accordion-item">
            <button class="accordion-button">Refresher: Foundation Models, PEFT, and LoRA</button>
            <div class="accordion-content">
                <p>For definitions of Foundation Models, PEFT, and LoRA, see the earlier accordion 
                <em>"What are Foundation Models, PEFT, and LoRA?"</em> in the <strong>Recent Advances</strong> section.</p>
            </div>
        </div>

        <blockquote>
            <p>Think of it as educating a highly experienced "AI Geoscientist" by having it study textbooks covering geology and wave physics from all around the globe (the pre-training on diverse data).</p>
            <p>When faced with data from a specific, new region (the downstream task), you don't need to reteach the AI everything from scratch. Instead, you provide it with a small, targeted "briefing note" specific to that region. This adaptation process is done using <strong class="key-concept">Parameter-Efficient Fine-Tuning (PEFT)</strong> techniques. <strong class="key-concept">Low-Rank Adaptation (LoRA)</strong> is a prominent PEFT method that achieves this adaptation by adding and training only a tiny number of new parameters, leaving the vast knowledge base of the original large model untouched (frozen).</p>
        </blockquote>

        <h4>Key Contributions</h4>
        <ul>
            <li>Demonstrates the feasibility and effectiveness of pre-training large-scale foundation models (e.g., based on U-Net architectures) for FWI using diverse, multi-domain datasets like combinations of OpenFWI families.</li>
            <li>Shows that PEFT methods, specifically LoRA, can successfully adapt these large pre-trained FWI models to new, specific target datasets (e.g., a single geological style not heavily represented in pre-training) by training only a minuscule fraction (e.g., <1%) of the total model parameters.</li>
            <li>Achieves performance comparable to, and sometimes even better than, fully fine-tuning the entire large model on the target task, especially for <span class="key-concept">out-of-distribution</span> generalization (performing well on data types different from the pre-training data). This comes with significant reductions in computational cost (GPU memory, training time) and storage requirements for adapted models. LoRA seems to act as a regularizer, preventing the model from overfitting to the smaller target dataset during adaptation.</li>
        </ul>
         <div class="accordion-item">
            <button class="accordion-button">What is Out-of-Distribution (OOD) Generalization?</button>
            <div class="accordion-content">
                 <p>Imagine training an AI to recognize pictures of cats and dogs. In-distribution generalization means it can recognize new pictures of cats and dogs it hasn't seen before. Out-of-distribution (OOD) generalization is harder â€“ it means the AI can still perform reasonably well when shown something different, perhaps a picture of a fox or a wolf, or maybe a cartoon cat instead of a photo. In FWI, OOD generalization means the model performs well on seismic data from a geological setting significantly different from those it was primarily trained on.</p>
                 <h5>More Rigorous Definition</h5>
                 <p>Out-of-Distribution (OOD) generalization refers to a model's ability to perform well on data drawn from a different distribution than the training data distribution. While standard generalization assesses performance on unseen data from the *same* distribution (test set), OOD generalization tests robustness to shifts in data characteristics, such as different noise levels, acquisition parameters, or, in FWI, substantially different geological environments or structures not well-represented in the training set. Achieving good OOD performance is a key challenge in deploying ML models in real-world scenarios.</p>
            </div>
        </div>

        <h4>Architecture & LoRA Method</h4>
        <p>
            The foundation model itself is often a large CNN, such as a U-Net, pre-trained on the task of mapping seismic data to velocity models using a large dataset. The magic happens during the adaptation phase with LoRA. Instead of retraining the existing weight matrices (e.g., in convolutional or linear layers) of the pre-trained model, LoRA introduces pairs of small, new, trainable matrices alongside the frozen original weights.
        </p>
        <p>
            Specifically, for an original weight matrix <span class="math-notation">Wâ‚€ âˆˆ â„�<sup class="superscript">dÃ—k</sup></span>, LoRA adds a low-rank update computed as <span class="math-notation">BA</span>, where <span class="math-notation">A âˆˆ â„�<sup class="superscript">rÃ—k</sup></span> and <span class="math-notation">B âˆˆ â„�<sup class="superscript">dÃ—r</sup></span>. Here, <span class="math-notation">r</span> is the "rank" of the adaptation, and crucially, <span class="math-notation">r â‰ª min(d, k)</span>. The forward pass becomes <span class="math-notation">h = Wâ‚€x + Î±(BA)x</span>, where <span class="math-notation">x</span> is the input, <span class="math-notation">h</span> is the output, and <span class="math-notation">Î±</span> is a scaling factor (often set as <span class="math-notation">Î±/r</span> in practice). During fine-tuning, only the matrices <span class="math-notation">A</span> and <span class="math-notation">B</span> are trained, while the large <span class="math-notation">Wâ‚€</span> remains frozen. Since <span class="math-notation">r</span> is small (e.g., 4, 8, or 16), the number of trainable parameters in <span class="math-notation">A</span> and <span class="math-notation">B</span> (<span class="math-notation">r*(d+k)</span>) is much smaller than in <span class="math-notation">Wâ‚€</span> (<span class="math-notation">d*k</span>). For convolutional layers, a similar adaptation is applied using low-rank convolutional kernels.
        </p>
        <div class="accordion-item">
            <button class="accordion-button">Intuition behind Low-Rank Matrices in LoRA</button>
            <div class="accordion-content">
                 <p>Think of the original large weight matrix <span class="math-notation">Wâ‚€</span> as containing a vast amount of general knowledge. When adapting to a new task, we assume the necessary *changes* or *adjustments* (<span class="math-notation">Î”W</span>) to this knowledge are relatively simple or structured, not requiring completely new, complex knowledge.</p>
                 <p>A low-rank matrix is mathematically "simpler" than a full-rank matrix of the same size. It essentially captures information that can be represented by combining a smaller number of basis vectors. By representing the update <span class="math-notation">Î”W</span> as the product of two skinny matrices (<span class="math-notation">B</span> and <span class="math-notation">A</span>), LoRA hypothesizes that the task-specific adaptation lies in a low-dimensional subspace. This allows it to learn the necessary adjustments with far fewer parameters than changing the entire <span class="math-notation">Wâ‚€</span> matrix.</p>
                 <p>The rank <span class="math-notation">r</span> controls the complexity (and number of parameters) of the adaptation. A very small <span class="math-notation">r</span> means a very simple adaptation, while a larger <span class="math-notation">r</span> allows for more complex adjustments but requires training more parameters.</p>
            </div>
        </div>

        <h4>Training Setup</h4>
        <ol>
            <li><strong>Pre-training:</strong> Train a large FWI model (e.g., a deep U-Net) on a diverse collection of datasets (e.g., 10 combined subsets from OpenFWI, as done in the paper) using a standard supervised loss (like MSE). This step is computationally intensive but done only once.</li>
            <li><strong>Adaptation (PEFT with LoRA):</strong> For a specific downstream task (e.g., inverting data from only the 'Fault' dataset family, or potentially a new real dataset):
                <ul>
                    <li>Load the pre-trained foundation model weights and freeze them (set `requires_grad=False`).</li>
                    <li>Inject LoRA adapter layers (pairs of low-rank matrices/kernels) next to the chosen layers of the frozen model (e.g., convolutional layers in the U-Net). Initialize matrix A randomly and matrix B with zeros.</li>
                    <li>Train *only* the parameters within these newly added LoRA adapter layers using the smaller target dataset and the same loss function.</li>
                </ul>
            </li>
        </ol>

        <h4>Reproducibility</h4>
        <p>
            The methodology relies on standard deep learning frameworks (like PyTorch) and the LoRA technique. Libraries facilitating PEFT, such as Hugging Face's `peft` library, provide readily usable implementations of LoRA for various layer types (Linear, Conv2d). Implementing this approach requires having a pre-trained FWI model and integrating these PEFT library components to perform the LoRA-based fine-tuning. The authors provide significant detail on their model architecture, pre-training setup, and LoRA configuration in their paper, aiding reproducibility.
        </p>

        <h4>Conceptual Implementation (PyTorch): LoRA for a Convolutional Layer</h4>
        <p>This snippet shows how a standard <code>nn.Conv2d</code> layer can be wrapped with LoRA adapters using a hypothetical <code>LoRAConv2d</code> class (implementations exist in libraries like <code>peft</code>).</p>
        <div class="code-container">
            <button class="toggle-code-btn">Toggle Code</button>
            <button class="copy-code-btn" title="Copy code">Copy</button>
            <pre class="line-numbers"><code class="language-python">
import torch
import torch.nn as nn
import math

# Note: This is a simplified illustrative implementation. Libraries like `peft` offer robust versions.
class LoRAConv2d(nn.Module):
    """Applies Low-Rank Adaptation to a nn.Conv2d layer."""
    def __init__(
        self,
        base_conv: nn.Conv2d,
        rank: int = 8,         # The rank 'r' of the adaptation
        alpha: float = 8.0,    # Scaling factor (often = rank)
        dropout: float = 0.0,  # Dropout for LoRA layers
    ):
        super().__init__()
        assert rank > 0
        self.base_conv = base_conv
        self.rank = rank
        self.alpha = alpha
        # Practical scaling used in many implementations
        self.scaling = self.alpha / self.rank

        # Freeze the original convolutional layer
        self.base_conv.weight.requires_grad = False
        if self.base_conv.bias is not None:
            self.base_conv.bias.requires_grad = False

        in_channels = base_conv.in_channels
        out_channels = base_conv.out_channels
        kernel_size = base_conv.kernel_size
        stride = base_conv.stride
        padding = base_conv.padding
        dilation = base_conv.dilation
        groups = base_conv.groups

        # LoRA A: Convolution down to rank `r`. Mimics the original conv's spatial operation.
        self.lora_A = nn.Conv2d(
            in_channels, self.rank * groups, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False
        )

        # LoRA B: Pointwise (1x1) conv up from rank `r` to original out_channels
        self.lora_B = nn.Conv2d(
            self.rank * groups, out_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False
        )

        self.lora_dropout = nn.Dropout(p=dropout)

        # Initialize weights: A with Kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base_conv(x)

        x_dropout = self.lora_dropout(x)
        lora_A_output = self.lora_A(x_dropout)
        lora_B_output = self.lora_B(lora_A_output)
        lora_update = lora_B_output * self.scaling

        return base_output + lora_update

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_conv.train(False) # Keep base layer frozen

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = {}
        destination[prefix + 'lora_A.weight'] = self.lora_A.weight.data if not keep_vars else self.lora_A.weight
        destination[prefix + 'lora_B.weight'] = self.lora_B.weight.data if not keep_vars else self.lora_B.weight
        return destination

    def load_state_dict(self, state_dict, strict: bool = True):
        prefix = self._get_prefix_from_load_state_dict_migration(prefix)
        missing_keys, unexpected_keys = super().load_state_dict(state_dict, strict=False)

        # Filter out expected missing keys for base_conv 
        expected_missing = {
            prefix + 'base_conv.weight',
            prefix + ('base_conv.bias' if self.base_conv.bias is not None else '')
        }
        expected_missing = {k for k in expected_missing if k}

        final_missing_keys = [k for k in missing_keys if k not in expected_missing]
        final_unexpected_keys = unexpected_keys

        if strict and (final_missing_keys or final_unexpected_keys):
             raise RuntimeError(
                 f"Error loading state_dict for {self.__class__.__name__}: "
                 f"missing keys {final_missing_keys}, unexpected keys {final_unexpected_keys}"
             )
        return torch.nn.modules.module._IncompatibleKeys(final_missing_keys, final_unexpected_keys)


# --- Example Usage ---
original_conv_layer = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
for param in original_conv_layer.parameters():
    param.requires_grad = False

lora_adapted_layer = LoRAConv2d(original_conv_layer, rank=4, alpha=4.0, dropout=0.1)

print("Trainable parameters in LoRA-adapted layer:")
trainable_params = []
for name, param in lora_adapted_layer.named_parameters():
    if param.requires_grad:
        trainable_params.append(name)
print(trainable_params)

input_tensor = torch.randn(4, 32, 128, 128)
output_tensor = lora_adapted_layer(input_tensor)
print(f"\nOutput shape: {output_tensor.shape}")
            </code></pre>
        </div>
        <p>
            This code defines a <code>LoRAConv2d</code> module that encapsulates a frozen base convolutional layer and adds the trainable low-rank adapter pathway (<code>lora_A</code>, <code>lora_B</code>). By strategically replacing standard layers in a large pre-trained network with these LoRA-enhanced versions, researchers can perform parameter-efficient fine-tuning, adapting powerful foundation models to new tasks with minimal computational overhead.
        </p>

        <h2 id="conclusion">Conclusion: Towards Smarter, Faster, and More Reliable FWI</h2>

        <p>
            The period from 2022 to 2025 has witnessed remarkable progress in applying deep learning to the complex challenges of Full Waveform Inversion. The field has matured beyond initial feasibility demonstrations to tackle fundamental issues like generalization to unseen geology, accurate recovery of fine details, robust handling of noisy data, and the principled integration of wave physics.
        </p>
        <ul>
            <li>
                <strong class="key-concept">More Sophisticated Architectures:</strong> We've seen the adoption of powerful architectures like Transformers (e.g., SVIT) capable of capturing global wavefield characteristics, complemented by specialized multi-stage networks (e.g., VIF-Net) designed to enhance specific aspects like interface sharpness.
            </li>
            <li>
                <strong class="key-concept">Deeper Physics Integration:</strong> Physics-informed approaches are gaining significant traction. Methods like IFWI (using neural networks as implicit representations of the velocity field) and FWI-GAN (leveraging adversarial learning constrained by physics simulation) reduce the dependence on vast labeled datasets and show promise in overcoming long-standing FWI challenges like cycle-skipping.
            </li>
            <li>
                <strong class="key-concept">Leveraging Data and Efficiency:</strong> The availability of large-scale, diverse benchmark datasets (OpenFWI) has been instrumental in driving progress and enabling rigorous model comparison. Concurrently, the emergence of FWI foundation models combined with parameter-efficient adaptation techniques like LoRA points towards a future where extremely powerful, general-purpose AI models can be rapidly and economically tailored for specific seismic surveys and geological targets, boosting both robustness and practical applicability.
            </li>
        </ul>
         <div class="accordion-item">
            <button class="accordion-button">FAQ: Which DL method is best for FWI?</button>
            <div class="accordion-content">
                 <p>There's no single "best" method; the optimal choice depends on the specific problem, available data, computational resources, and goals:</p>
                 <ul>
                     <li><strong>For speed after training and if large labeled datasets are available:</strong> Supervised methods like U-Nets (e.g., InversionNet) or Transformers (e.g., SVIT) are strong candidates. Foundation models with PEFT offer efficiency if a good pre-trained model exists.</li>
                     <li><strong>If labeled velocity maps are scarce or unavailable, or robustness to cycle-skipping is paramount:</strong> Physics-informed methods like IFWI or FWI-GAN are attractive, as they learn directly from observed seismic data and physical principles.</li>
                     <li><strong>For resolving sharp interfaces:</strong> Specialized architectures like VIF-Net might outperform general-purpose models.</li>
                     <li><strong>For handling diverse data types or complex relationships:</strong> Transformers might offer advantages due to their global attention mechanism.</li>
                 </ul>
                 <p>Often, hybrid approaches combining strengths (e.g., using a DL model for a good starting guess for traditional FWI, or incorporating physical constraints into a supervised network) show significant promise.</p>
            </div>
        </div>

        <p>
            The trajectory suggests that the most fruitful path forward likely involves synergistic combinations: harnessing the pattern-recognition capabilities of advanced deep learning architectures, grounding them in the fundamental laws of wave physics, training them on diverse large-scale data, and utilizing efficient adaptation strategies for deployment. While significant challenges undoubtedly remainâ€”particularly concerning the application to highly complex, noisy, and large-scale 3D field dataâ€”the pace of innovation is rapid. AI-driven FWI is steadily evolving into an indispensable tool in the geoscientist's toolkit for imaging and understanding the Earth's intricate subsurface.
        </p>

        <div class="sources" id="sources">
            <h2>Sources & Further Reading</h2>
            <ul>
                {/* Merged and slightly deduplicated list */}
                <li>Deng, Chen, et al. (2022). <strong>OpenFWI: Large-Scale Multi-Structural Benchmark Datasets for Seismic Full Waveform Inversion</strong>. Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track. (<a href="https://github.com/lanl/OpenFWI" target="_blank" rel="noopener">GitHub</a>)</li>
                <li>Maiti, D., et al. (2024). <strong>Parameter Efficient Fine-Tuning Approach for Deep Learning Based Seismic Full Waveform Inversion</strong>. arXiv preprint arXiv:2402.19510. (<a href="https://arxiv.org/abs/2402.19510" target="_blank" rel="noopener">arXiv</a>)</li>
                <li>Li, Y., & Yang, B. (2023). <strong>Seismic velocity inversion transformer</strong>. GEOPHYSICS, 88(4), R513â€“R533. (<a href="https://library.seg.org/doi/10.1190/geo2022-0551.1" target="_blank" rel="noopener">SEG Library</a>)</li>
                <li>Wang, Z., et al. (2024). <strong>Deep learning for interface completion constrained full-waveform inversion based on velocity interface fusion network</strong>. Computers & Geosciences, 174, 105834. (<a href="https://www.sciencedirect.com/science/article/abs/pii/S009830042400054X" target="_blank" rel="noopener">ScienceDirect</a>)</li>
                <li>Sun, J., & Innanen, K. A. (2023). <strong>Implicit Full-Waveform Inversion With Deep Neural Representation</strong>. Journal of Geophysical Research: Solid Earth, 128(4), e2022JB025900. (Preprint: <a href="https://arxiv.org/abs/2209.03525" target="_blank" rel="noopener">arXiv</a>, Journal: <a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2022JB025900" target="_blank" rel="noopener">JGR</a>)</li>
                <li>Yang, F., & Ma, J. (2023). <strong>FWIGAN: Full-Waveform Inversion via a Physics-Informed Generative Adversarial Network</strong>. Journal of Geophysical Research: Solid Earth, 128(4), e2022JB025947. (<a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2022JB025947" target="_blank" rel="noopener">JGR</a>, Code: <a href="https://github.com/YangFangShu/FWIGAN" target="_blank" rel="noopener">GitHub</a>)</li>
                <li>Jin, B., et al. (2024). <strong>An empirical study of large-scale data-driven seismic full-waveform inversion</strong>. Scientific Reports, 14(1), 20034. (<a href="https://www.nature.com/articles/s41598-024-70745-1" target="_blank" rel="noopener">Nature Scientific Reports</a>)</li>
                <li>Li, J., et al. (2023). <strong>TransInver: High-resolution 3-D seismic full-waveform inversion based on the self-attention mechanism</strong>. GEOPHYSICS, 88(6), R799-R817. (Advanced 3D Transformer FWI Example) (<a href="https://library.seg.org/doi/10.1190/geo2023-0048.1" target="_blank" rel="noopener">SEG Library</a>)</li>
                <li>Chen, X., et al. (2023). <strong>DiffusionVel: Generative Diffusion Model based Seismic Velocity Inversion</strong>. arXiv preprint arXiv:2309.17309. (Diffusion models approach) (<a href="https://arxiv.org/abs/2309.17309" target="_blank" rel="noopener">arXiv</a>)</li>
                <li>Vaswani, A., et al. (2017). <strong>Attention is All You Need</strong>. Advances in Neural Information Processing Systems (NeurIPS). (Original Transformer Paper) (<a href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noopener">arXiv</a>)</li>
                <li>Ronneberger, O., Fischer, P., & Brox, T. (2015). <strong>U-Net: Convolutional Networks for Biomedical Image Segmentation</strong>. MICCAI. (Original U-Net Paper) (<a href="https://arxiv.org/abs/1505.04597" target="_blank" rel="noopener">arXiv</a>)</li>
                 <li>Goodfellow, I. et al. (2014). <strong>Generative Adversarial Nets</strong>. Advances in Neural Information Processing Systems (NeurIPS). (Original GAN Paper) (<a href="https://arxiv.org/abs/1406.2661" target="_blank" rel="noopener">arXiv</a>)</li>
                <li>Arjovsky, M., Chintala, S., & Bottou, L. (2017). <strong>Wasserstein GAN</strong>. arXiv preprint arXiv:1701.07875. (<a href="https://arxiv.org/abs/1701.07875" target="_blank" rel="noopener">arXiv</a>)</li>
                <li>Gulrajani, I., et al. (2017). <strong>Improved Training of Wasserstein GANs</strong>. Advances in Neural Information Processing Systems (NeurIPS). (WGAN-GP Paper) (<a href="https://arxiv.org/abs/1704.00028" target="_blank" rel="noopener">arXiv</a>)</li>
                <li>Hu, E. J., et al. (2021). <strong>LoRA: Low-Rank Adaptation of Large Language Models</strong>. International Conference on Learning Representations (ICLR). (Original LoRA Paper) (<a href="https://arxiv.org/abs/2106.09685" target="_blank" rel="noopener">arXiv</a>)</li>
                <li>Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). <strong>Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations</strong>. Journal of Computational Physics, 378, 686-707. (Key PINN Paper) (<a href="https://www.sciencedirect.com/science/article/pii/S002199911830662X" target="_blank" rel="noopener">Journal</a>)</li>
                 <li>Virieux, J., & Operto, S. (2009). <strong>An overview of full-waveform inversion in exploration geophysics</strong>. Geophysics, 74(6), WCC1-WCC26. (Classic FWI Review) (<a href="https://library.seg.org/doi/10.1190/1.3238367" target="_blank" rel="noopener">SEG Library</a>)</li>
            </ul>
        </div>


        <!-- ======================================= -->
        <!-- == NEW PREVIOUS COMPETITIONS SECTION == -->
        <!-- ======================================= -->
        <div class="previous-competitions" id="previous-competitions">
            <h2>Previous Competitions Review (Insights for FWI)</h2>
            <p>
                Learning from past data science competitions that involved complex signals, inverse problems, or geophysical data can provide valuable strategies for the current FWI challenge. Here's a summary of insights, ordered by relevance:
            </p>

            <h3 id="prev-openfwi">Full Waveform Inversion Benchmarks (OpenFWI, 2021)</h3>
            <ul>
                <li>Directly relevant: Provided benchmark datasets (synthetic seismic + velocity models) for ML-based FWI, similar to this competition's goal.</li>
                <li>Common Architecture: U-Net (CNN) architectures proved effective for the image-to-image task of mapping seismic data to velocity maps.</li>
                <li>Hybrid Models: Combining traditional physics-based inversion (e.g., for an initial model) with ML refinement often outperformed purely data-driven approaches.</li>
                <li>Multi-Scale Training: Mimicking classical FWI, training ML models progressively from low to high frequencies helped stability and accuracy.</li>
                <li>Data Diversity & Augmentation: Showcased the need for diverse geological scenarios in training data for generalization. Augmentation (noise, shifts, trace drops) improved robustness.</li>
            </ul>

            <h3 id="prev-seg-facies">SEG AI Challenge: Seismic Facies Identification (2022)</h3>
            <ul>
                <li>Relevant Domain: Focused on interpreting seismic images (post-stack data) to identify geological facies (segmentation).</li>
                <li>Dominant Architecture: U-Net CNN variants were heavily used and successful for pixel-wise classification/segmentation of seismic images.</li>
                <li>Ensembling: Top solutions frequently used ensembles of multiple U-Nets to improve robustness and generalization.</li>
                <li>Augmentation: Heavy data augmentation (noise, phase shifts, distortions) was crucial for handling real-world seismic complexities.</li>
                <li>Post-processing: Applying geological constraints (smoothing, filtering small patches) post-prediction improved the realism of results.</li>
                <li>Class Imbalance: Techniques to handle imbalanced facies data (e.g., weighted loss) might be relevant for FWI (e.g., weighting deeper features).</li>
            </ul>

            <h3 id="prev-tgs-salt">TGS Salt Identification Challenge (Kaggle 2018)</h3>
            <ul>
                <li>Relevant Domain: Similar to SEG Facies, focused on segmenting salt bodies in seismic amplitude images.</li>
                <li>Dominant Architecture: U-Net CNNs were the core of all top solutions, confirming their suitability for seismic image interpretation.</li>
                <li>Key Techniques: Introduced useful tricks like LovÃ¡sz hinge loss (optimizing IoU directly) and test-time augmentation (averaging predictions from augmented inputs) for better performance.</li>
                <li>Post-processing: Emphasized cleaning up predicted boundaries, analogous to enforcing smoothness or known constraints in FWI velocity models.</li>
            </ul>

            <h3 id="prev-fastmri">FastMRI Image Reconstruction Challenge (Facebook NYU 2020)</h3>
            <ul>
                <li>Relevant Problem Type: An inverse problem (image reconstruction from under-sampled frequency-domain data), analogous to FWI recovering velocity from wave data.</li>
                <li>Physics-Guided AI: Winning solutions used physics-guided neural networks, embedding NNs within iterative physics-based reconstruction algorithms (unrolled optimization).</li>
                <li>Hybrid Methods: Showcased success by alternating between applying the physical forward model (MRI simulation) and a learned (CNN) correction step. This suggests potential for similar hybrid physics/ML cycles in FWI.</li>
                <li>Data Consistency: Enforcing that the output prediction, when run through the forward model, matches the original measurements improved results.</li>
                <li>Architecture: U-Nets were also used here for mapping sensor data to images.</li>
            </ul>

            <h3 id="prev-ventilator">Kaggle Ventilator Pressure Prediction (2021)</h3>
            <ul>
                <li>Relevant Problem Type: Inverse problem (inferring internal pressure from time-series signals) using simulated, physics-based data.</li>
                <li>Sequence Modeling: Recurrent Neural Networks (RNNs, specifically LSTMs) were effective for capturing temporal dependencies in the waveform-like data.</li>
                <li>Physics Integration: Blending ML with physics knowledge (e.g., custom loss functions penalizing unphysical predictions, using physics equations) improved generalization.</li>
                <li>Ensembling: Ensembles of different models (e.g., LSTMs + Gradient Boosting) or across folds were crucial for top performance.</li>
                <li>Robustness: Techniques like robust loss functions (Huber loss) helped handle outliers or abrupt signal changes.</li>
            </ul>

            <h3 id="prev-radiant-earth">Radiant Earth Tropical Storm Wind Speed (DrivenData 2021)</h3>
            <ul>
                <li>Relevant Data Type: Spatio-temporal signal processing (sequences of satellite images) to predict a physical quantity (wind speed).</li>
                <li>Combined Architectures: CNNs (for spatial features per image) combined with sequence models like LSTMs (for temporal evolution) were highly successful.</li>
                <li>Heavy Ensembling: The winning solution ensembled 51 diverse models, highlighting the power of model diversity for complex signal interpretation.</li>
                <li>Heavy Augmentation: Extensive augmentation (rotation, flipping, time-shifts, scaling) was critical for model invariance and generalization.</li>
                <li>Domain Knowledge: Although no explicit physics, successful models implicitly learned visual cues known to meteorologists (Dvorak technique).</li>
            </ul>

            <h3 id="prev-lanl-earthquake">LANL Earthquake Prediction (Kaggle 2019)</h3>
            <ul>
                <li>Relevant Data Type: Directly used seismic time-series data (acoustic emissions).</li>
                <li>Signal Processing Features: Top solutions often relied on engineered features capturing signal statistics and frequency content changes, fed into boosting models.</li>
                <li>CNNs for Waveforms: 1D CNNs applied directly to raw waveforms were also effective.</li>
                <li>Frequency Analysis: Highlighted the importance of analyzing how frequency content changes over time, potentially relevant for analyzing different phases or features in FWI waveforms.</li>
            </ul>

            <h3 id="prev-weather4cast">Weather4Cast Spatiotemporal Forecasting (NeurIPS 2022)</h3>
            <ul>
                <li>Relevant Task: Forecasting the evolution of a physical field (precipitation radar) over time.</li>
                <li>Advanced Sequence Models: Top teams used Convolutional LSTMs and Transformers to model spatio-temporal dynamics.</li>
                <li>Auxiliary Input: Incorporating hints about motion (like optical flow) improved predictions, suggesting that providing initial velocity guesses or travel times might help FWI models.</li>
                <li>Transformers/Attention: Suggests potential for using attention mechanisms to focus FWI models on specific parts of the waveform data (e.g., first arrivals vs. reflections).</li>
            </ul>
        </div>
        <!-- ======================================= -->
        <!-- == END PREVIOUS COMPETITIONS SECTION == -->
        <!-- ======================================= -->

    </div> <!-- End Main Content -->

</div> <!-- End Container -->

<!-- Place this script block right at the very end of the %%html cell -->
<!-- Scripts for Interactivity and Syntax Highlighting -->
<!-- Prism.js Core -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-core.min.js"></script>
<!-- Prism.js Autoloader (loads languages based on class name) -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/autoloader/prism-autoloader.min.js"></script>
<!-- Prism.js Line Numbers Plugin -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/line-numbers/prism-line-numbers.min.js"></script>

<script>
    // --- Run immediately after the HTML in the cell is parsed ---
    console.log("Executing script within %%html cell.");

    try {
        // --- Find the main container for this cell's content ---
        const containerId = 'fwi-review-cell-content-123'; // <<< IMPORTANT: Match the ID you added above
        const mainContainer = document.getElementById(containerId);

        if (!mainContainer) {
            console.error(`Main container with ID #${containerId} not found. Scripts cannot run.`);
            // Add class to body to potentially hide JS-dependent elements via CSS
            document.body.classList.add('no-js');
        } else {
            console.log(`Main container #${containerId} found.`);

            // --- 1. Initialize Prism.js Syntax Highlighting (Scoped) ---
            if (typeof Prism !== 'undefined') {
                console.log("Prism object found. Highlighting within main container.");
                // Highlight only elements *within* this specific container
                Prism.highlightAllUnder(mainContainer);
                console.log("Prism highlighting finished within container.");
            } else {
                console.error("Prism object is undefined. Check if Prism scripts loaded correctly.");
            }

            // --- 2. Add Code Block Toggling (Scoped) ---
            const codeContainers = mainContainer.querySelectorAll('.code-container');
            console.log(`Found ${codeContainers.length} code containers within #${containerId}.`);
            codeContainers.forEach((container, index) => {
                const button = container.querySelector('.toggle-code-btn');
                const preElement = container.querySelector('pre');

                if (button && preElement) {
                     // Start with code collapsed
                    if (!container.classList.contains('code-collapsed')) {
                        container.classList.add('code-collapsed');
                    }
                    button.textContent = 'Show Code'; // Set initial text

                    button.addEventListener('click', () => {
                        console.log(`Code toggle button ${index+1} clicked.`);
                        container.classList.toggle('code-collapsed');
                        button.textContent = container.classList.contains('code-collapsed') ? 'Show Code' : 'Hide Code';
                    });
                } else {
                    console.warn(`Code container ${index+1} missing button or pre element.`);
                }
            });

            // --- 3. Add Accordion Toggling (Scoped & REFINED) ---
            const accordionButtons = mainContainer.querySelectorAll('.accordion-button');
            console.log(`Found ${accordionButtons.length} accordion buttons within #${containerId}.`);
            accordionButtons.forEach((button, index) => {
                const content = button.nextElementSibling;

                if (button && content && content.classList.contains('accordion-content')) {
                    button.addEventListener('click', () => {
                        console.log(`Accordion button ${index+1} ('${button.textContent.trim().substring(0,20)}...') clicked.`);
                        const isActive = button.classList.contains('active');

                        // Toggle the current accordion
                        button.classList.toggle('active', !isActive);
                        content.classList.toggle('show', !isActive);

                        // Set max-height for transition
                        if (!isActive) { // If opening
                             // Use rAF to ensure 'show' class is applied and element is visible
                             // before calculating scrollHeight
                             requestAnimationFrame(() => {
                                content.style.maxHeight = content.scrollHeight + "px";
                                console.log(`Opening accordion ${index+1}, setting maxHeight: ${content.style.maxHeight}`);
                             });
                        } else { // If closing
                            // Reset maxHeight to null to let CSS transition back to 0
                            content.style.maxHeight = null;
                            console.log(`Closing accordion ${index+1}, resetting maxHeight.`);
                        }

                        // Optional: Close others logic (uncomment if needed)
                        /*
                        mainContainer.querySelectorAll('.accordion-button').forEach(otherButton => {
                            if (otherButton !== button && otherButton.classList.contains('active')) {
                                 console.log(`Closing other accordion: '${otherButton.textContent.trim().substring(0,20)}...'`);
                                 otherButton.classList.remove('active');
                                 const otherContent = otherButton.nextElementSibling;
                                 if(otherContent && otherContent.classList.contains('accordion-content')) {
                                     otherContent.classList.remove('show');
                                     otherContent.style.maxHeight = null;
                                 }
                            }
                        });
                        */
                    });
                } else {
                    console.warn(`Accordion button ${index+1} missing valid content sibling or content missing .accordion-content class.`);
                }
            });

            // --- 4. Add Code Copy Button Functionality (Scoped) ---
            const copyButtons = mainContainer.querySelectorAll('.copy-code-btn');
            console.log(`Found ${copyButtons.length} copy buttons within #${containerId}.`);
            copyButtons.forEach(button => {
                button.addEventListener('click', () => {
                    const codeContainer = button.closest('.code-container');
                    if (!codeContainer) return; // Should not happen with current structure

                    const preElement = codeContainer.querySelector('pre');
                    const codeElement = preElement ? preElement.querySelector('code') : null;

                    if (codeElement) {
                        const codeToCopy = codeElement.textContent || "";
                        navigator.clipboard.writeText(codeToCopy).then(() => {
                            // Success feedback
                            button.textContent = 'Copied!';
                            button.classList.add('copied');
                            console.log('Code copied to clipboard.');
                            // Reset button text after a delay
                            setTimeout(() => {
                                button.textContent = 'Copy';
                                button.classList.remove('copied');
                            }, 2000); // Reset after 2 seconds
                        }).catch(err => {
                            // Error feedback (less common with modern browsers)
                            button.textContent = 'Error';
                            console.error('Failed to copy code: ', err);
                             setTimeout(() => {
                                button.textContent = 'Copy';
                             }, 2000);
                        });
                    } else {
                        console.warn('Could not find code element for copy button.');
                    }
                });
            });


        } // end if(mainContainer)

    } catch (error) {
        console.error("Error during script execution in %%html cell:", error);
        // Add class to body to potentially hide JS-dependent elements via CSS
         document.body.classList.add('no-js');
    }

    console.log("Script execution finished for %%html cell.");
</script>


</body>
</html>



# ================================================================
# 0. Imports & paths
# ================================================================
from pathlib import Path
import re, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

ROOT_TRAIN = Path("/kaggle/input/waveform-inversion/train_samples")
ROOT_TEST  = Path("/kaggle/input/waveform-inversion/test")

# ------------------------------------------------
# helper: derive the numeric suffix key
# ------------------------------------------------
def suffix_key(stem: str) -> str:
    """
    Strip the family prefix (data|seis|model|vel) -> return the rest.
    data2   -> 2
    seis4_1 -> 4_1
    """
    return re.sub(r"^(data|seis|model|vel)", "", stem)

# ================================================================
# 1. Robust loader: pair seismic â†” velocity within every scenario
# ================================================================
seis_blocks, vel_blocks = [], []

for data_dir in tqdm(sorted(ROOT_TRAIN.glob("**/data")), desc="Scanning scenarios"):
    scen_dir  = data_dir.parent
    model_dir = scen_dir / "model"

    # accept both naming schemes
    data_files  = list(data_dir.glob("data*.npy"))  + list(data_dir.glob("seis*.npy"))
    model_files = list(model_dir.glob("model*.npy"))+ list(model_dir.glob("vel*.npy"))

    # build dicts keyed by suffix
    data_dict  = {suffix_key(p.stem): p for p in data_files}
    model_dict = {suffix_key(p.stem): p for p in model_files}

    missing_data   = model_dict.keys() - data_dict.keys()
    missing_model  = data_dict.keys()  - model_dict.keys()
    if missing_data or missing_model:
        raise RuntimeError(f"{scen_dir}\n"
                           f" missing data:  {sorted(missing_data)[:5]}\n"
                           f" missing model: {sorted(missing_model)[:5]}")

    for k in sorted(data_dict.keys()):
        seis_blocks.append(np.load(data_dict[k]))   # (500,5,1000,70)
        vel_blocks .append(np.load(model_dict[k]))  # (500,1,70,70)

seis_all = np.concatenate(seis_blocks, axis=0)
vel_all  = np.concatenate(vel_blocks , axis=0)

print(f"\nâœ…  Loaded {seis_all.shape[0]:,} paired samples")
print(f"Seis tensor : {seis_all.shape}  (N,S,T,R)")
print(f"Vel  tensor : {vel_all.shape}   (N,1,H,W)")
print(f"Seis  âˆˆ [{seis_all.min():.2f}, {seis_all.max():.2f}]")
print(f"Vel   âˆˆ [{vel_all.min():.1f}, {vel_all.max():.1f}]")

# peek at one example
plt.figure(figsize=(5,4))
plt.imshow(seis_all[0,0], aspect='auto', cmap='seismic')
plt.title("First source gather, sampleÂ 0")
plt.xlabel("Receiver"); plt.ylabel("Time"); plt.show()

plt.figure(figsize=(4,4))
plt.imshow(vel_all[0,0], cmap='viridis')
plt.title("Velocity map, sampleÂ 0"); plt.colorbar(); plt.show()

# ================================================================
# 2. Baseline inversion â€” global mean velocity
# ================================================================
mean_vel     = vel_all.mean(axis=0, keepdims=True)    # (1,1,70,70)
test_files   = sorted(ROOT_TEST.glob("*.npy"))
N_test       = len(test_files)
print(f"\nğŸ“�  {N_test:,} test seismograms detected")

predictions  = np.tile(mean_vel, (N_test,1,1,1))      # (N,1,70,70)

# ================================================================
# 3. Build submission.csv
# ================================================================
preds_2d = predictions.squeeze(1)                     # (N,70,70)

rows = []
for i, fpath in enumerate(test_files):
    sample_id = fpath.stem
    for y in range(70):
        rows.append([f"{sample_id}_y_{y}"] + preds_2d[i, y, 1::2].tolist())

columns = ["oid_ypos"] + [f"x_{x}" for x in range(1,70,2)]
pd.DataFrame(rows, columns=columns).to_csv("submission.csv", index=False)

print("ğŸš€  Saved submission.csv â€” shape:", len(rows), "rows Ã—", len(columns), "cols")


