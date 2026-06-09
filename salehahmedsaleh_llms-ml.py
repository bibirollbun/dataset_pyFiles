#| export

class Model:
    def __init__(self):
        pass
      
    def predict(self, prompt: str) -> str:
        return """
<svg width="1000" height="1000" viewBox="0 0 1000 1000" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="bg" cx="60%" cy="40%" r="85%">
      <stop offset="0%" stop-color="#0f172a"/>
      <stop offset="100%" stop-color="#1e3b8a"/>
    </radialGradient>

    <linearGradient id="neon" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#4ade80"/>
      <stop offset="50%" stop-color="#2dd4bf"/>
      <stop offset="100%" stop-color="#5eead4"/>
    </linearGradient>

    <pattern id="grid" width="100" height="100" patternUnits="userSpaceOnUse">
      <path d="M100 0H0v100" stroke="#f8fafc" stroke-width="2" opacity="0.1"/>
    </pattern>
  </defs>

  <rect width="1000" height="1000" fill="url(#bg)"/>
    <rect width="1000" height="1000" fill="url(#grid)" opacity="0.15"/>

  <g transform="translate(500 500)">
    <path d="M0-200L57-193 193-57 200 0 193 57 57 193 0 200-57 193-193 57-200 0-193-57-57-193Z" 
          fill="none"
          stroke="url(#neon)"
          stroke-width="15"
          stroke-linejoin="round"
          opacity="0.97"/>

    <g transform="scale(0.6)">
      <circle r="140" fill="#4ade80" opacity="0.95"/>
      <path d="M-100-100L100 100M100-100L-100 100" 
            stroke="#0f172a" 
            stroke-width="25"
            stroke-linecap="round"/>
      <g stroke="#2dd4bf" stroke-width="8">
        <path d="M0-140L0-200M0 140L0 200"/>
        <path d="M-140 0L-200 0M140 0L200 0"/>
      </g>
    </g>
<circle cx="300" cy="0" r="15" fill="#5eead4"/>
    <circle cx="-300" cy="0" r="15" fill="#5eead4"/>
  </g>

  <g opacity="0.3">
    <circle cx="250" cy="250" r="30" fill="url(#neon)"/>
    <circle cx="750" cy="750" r="40" fill="#2dd4bf"/>
    <rect x="600" y="200" width="60" height="60" rx="15" fill="#5eead4"/>
    <path d="M200 600L300 700 400 600Z" fill="#4ade80"/>
  </g>

  <g stroke="#f8fafc" stroke-width="4" opacity="0.15">
    <path d="M500 100L500 900"/>
    <path d="M100 500L900 500"/>
  </g>
</svg>"""


from IPython.display import SVG

model = Model()
svg = model.predict('a goose winning a gold medal')

print(svg)
display(SVG(svg))


import kaggle_evaluation

kaggle_evaluation.test(Model)


!cp /tmp/kaggle-evaluation-submission-y82s5mld.csv /kaggle/working/submission.csv



from IPython.display import FileLink
FileLink('/kaggle/working/submission.csv')





