pip install cairosvg


import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cairosvg

SVG_CODE = '''
<svg viewBox="0 0 400 300" xmlns="http://www.w3.org/2000/svg">
  <!-- 胴体 -->
  <ellipse cx="200" cy="180" rx="80" ry="40" fill="white" stroke="black" stroke-width="2" />
  
  <!-- 首 -->
  <path d="M260,140 Q280,100 310,120" fill="white" stroke="black" stroke-width="2" />
  
  <!-- 頭 -->
  <path d="M310,120 
           Q330,110 340,130 
           Q345,145 335,155 
           Q320,160 310,155 
           Q305,150 310,140 
           Q315,130 310,120" 
        fill="white" stroke="black" stroke-width="2" />
  
  <!-- 角 -->
  <polygon points="335,125 345,80 325,80" fill="gold" />
  
  <!-- 目 -->
  <circle cx="325" cy="140" r="3" fill="black" />
  
  <!-- たてがみ -->
  <path d="M305,130 Q295,110 285,115" fill="none" stroke="purple" stroke-width="2" />
  <path d="M305,135 Q295,115 285,120" fill="none" stroke="pink" stroke-width="2" />
  
  <!-- 尻尾 -->
  <path d="M120,180 Q100,200 110,220 Q120,240 140,230" fill="none" stroke="blue" stroke-width="2" />
  
  <!-- 脚 (前) -->
  <path d="M240,220 L240,260" stroke="black" stroke-width="4" />
  <path d="M260,220 L260,260" stroke="black" stroke-width="4" />
  
  <!-- 脚 (後ろ) -->
  <path d="M160,220 L160,260" stroke="black" stroke-width="4" />
  <path d="M180,220 L180,260" stroke="black" stroke-width="4" />
</svg>

'''

# SVGをPNGに変換し、バイトデータを取得
png_data = cairosvg.svg2png(bytestring=SVG_CODE.encode('utf-8'))

# BytesIOオブジェクトに変換
png_buffer = io.BytesIO(png_data)

# matplotlibで画像を読み込み
img = mpimg.imread(png_buffer, format='png')

# 画像を表示
plt.imshow(img)
plt.title("unicorn")
plt.axis('off')  # 軸を非表示に
plt.show()





