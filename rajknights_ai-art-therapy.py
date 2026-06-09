import React, { useState, useEffect } from 'react';
import { Heart, Palette, MessageCircle, Sparkles, Camera, Send, Menu } from 'lucide-react';

const ArtTherapyCompanion = () => {
  const [currentView, setCurrentView] = useState('home');
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionData, setSessionData] = useState({
    mood: null,
    artPrompt: null,
    currentExercise: null
  });
  const [generatedArt, setGeneratedArt] = useState(null);

  const moodOptions = [
    { emoji: 'ðŸ˜Š', label: 'Happy', color: 'bg-yellow-400' },
    { emoji: 'ðŸ˜Œ', label: 'Calm', color: 'bg-blue-400' },
    { emoji: 'ðŸ˜”', label: 'Sad', color: 'bg-gray-400' },
    { emoji: 'ðŸ˜°', label: 'Anxious', color: 'bg-purple-400' },
    { emoji: 'ðŸ˜¤', label: 'Frustrated', color: 'bg-red-400' },
    { emoji: 'ðŸ¤”', label: 'Confused', color: 'bg-orange-400' }
  ];

  const artExercises = [
    {
      id: 1,
      title: "Color & Emotion",
      description: "Explore abstract colors and shapes to express how you're feeling",
      prompt: "Let's explore colors together. What colors represent your current emotional state?"
    },
    {
      id: 2,
      title: "Mindful Observation",
      description: "View calming abstract art and reflect on your emotional response",
      prompt: "I'll show you some abstract art. Take a deep breath and tell me what feelings arise."
    },
    {
      id: 3,
      title: "Creative Expression",
      description: "Describe an image from your mind, and I'll help bring it to life",
      prompt: "Close your eyes for a moment. What image comes to mind when you think about your day?"
    },
    {
      id: 4,
      title: "Gratitude Visualization",
      description: "Transform positive thoughts into visual art",
      prompt: "Think of something you're grateful for today. Can you describe it to me?"
    }
  ];

  const abstractArtStyles = [
    "flowing watercolor waves in calming blues and purples, abstract, peaceful, therapeutic",
    "soft geometric shapes in warm sunset colors, minimalist, soothing, meditative",
    "gentle swirling patterns in pastel colors, dreamy, abstract expressionism",
    "organic flowing forms in nature-inspired greens and earth tones, calming, abstract",
    "soft light rays and gentle gradients in healing colors, peaceful, abstract art"
  ];

  const startSession = async (exercise) => {
    setCurrentView('chat');
    setSessionData({ ...sessionData, currentExercise: exercise });
    
    const welcomeMessage = {
      role: 'assistant',
      content: `Welcome to the ${exercise.title} exercise. ${exercise.prompt}\n\nTake your time, there's no rush. I'm here to listen and support you.`
    };
    
    setMessages([welcomeMessage]);
  };

  const generateAbstractArt = (userDescription) => {
    // Simulate art generation with a carefully crafted abstract image
    const randomStyle = abstractArtStyles[Math.floor(Math.random() * abstractArtStyles.length)];
    
    // Create a canvas-based abstract art
    const canvas = document.createElement('canvas');
    canvas.width = 800;
    canvas.height = 600;
    const ctx = canvas.getContext('2d');
    
    // Create gradient background
    const gradient = ctx.createLinearGradient(0, 0, 800, 600);
    const colors = getColorsFromMood(sessionData.mood);
    gradient.addColorStop(0, colors[0]);
    gradient.addColorStop(0.5, colors[1]);
    gradient.addColorStop(1, colors[2]);
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 800, 600);
    
    // Add abstract shapes
    for (let i = 0; i < 12; i++) {
      ctx.globalAlpha = 0.3 + Math.random() * 0.4;
      ctx.fillStyle = colors[Math.floor(Math.random() * colors.length)];
      
      const x = Math.random() * 800;
      const y = Math.random() * 600;
      const size = 50 + Math.random() * 150;
      
      if (Math.random() > 0.5) {
        ctx.beginPath();
        ctx.arc(x, y, size, 0, Math.PI * 2);
        ctx.fill();
      } else {
        ctx.fillRect(x, y, size, size);
      }
    }
    
    return canvas.toDataURL();
  };

  const getColorsFromMood = (mood) => {
    const colorPalettes = {
      'Happy': ['#FFD700', '#FFA500', '#FF6B6B'],
      'Calm': ['#4ECDC4', '#44A08D', '#A8E6CF'],
      'Sad': ['#95A3B3', '#6C7A89', '#B8C6DB'],
      'Anxious': ['#B19CD9', '#8E44AD', '#DDA0DD'],
      'Frustrated': ['#E74C3C', '#C0392B', '#FF6B6B'],
      'Confused': ['#F39C12', '#E67E22', '#FFB347']
    };
    return colorPalettes[mood] || ['#95E1D3', '#F38181', '#EAFFD0'];
  };

  const handleSendMessage = async () => {
    if (!userInput.trim() || isLoading) return;

    const userMessage = { role: 'user', content: userInput };
    setMessages(prev => [...prev, userMessage]);
    setUserInput('');
    setIsLoading(true);

    try {
      const response = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'claude-sonnet-4-20250514',
          max_tokens: 1000,
          messages: [
            {
              role: 'user',
              content: `You are a compassionate AI art therapy companion. You help users explore their emotions through art and mindful reflection. 

Current exercise: ${sessionData.currentExercise?.title}
User's mood: ${sessionData.mood || 'Not specified'}

Guidelines:
- Be warm, empathetic, and non-judgmental
- Ask open-ended questions about feelings and emotions
- Encourage self-expression through art descriptions
- Validate the user's feelings
- Keep responses concise (2-3 sentences)
- If the user describes imagery or emotions, offer to generate abstract art for them
- Use gentle, supportive language

Conversation history:
${messages.slice(-4).map(m => `${m.role}: ${m.content}`).join('\n')}

User's message: ${userInput}

Respond as the art therapy companion:`
            }
          ]
        })
      });

      const data = await response.json();
      const assistantMessage = {
        role: 'assistant',
        content: data.content[0].text
      };

      setMessages(prev => [...prev, assistantMessage]);

      // Check if we should generate art based on the conversation
      if (messages.length > 2 && Math.random() > 0.4) {
        setTimeout(() => {
          const artData = generateAbstractArt(userInput);
          setGeneratedArt(artData);
        }, 1000);
      }

    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: "I'm here with you. Sometimes I might have trouble connecting, but your feelings are valid and important. Would you like to try describing what you're experiencing again?"
      }]);
    }

    setIsLoading(false);
  };

  const handleMoodSelect = (mood) => {
    setSessionData({ ...sessionData, mood: mood.label });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-pink-50 to-blue-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-gradient-to-r from-purple-500 to-pink-500 p-2 rounded-lg">
              <Palette className="text-white" size={24} />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-800">Art-Therapy Companion</h1>
              <p className="text-xs text-gray-500">Your creative wellness journey</p>
            </div>
          </div>
          <button
            onClick={() => setCurrentView('home')}
            className="p-2 hover:bg-gray-100 rounded-lg transition"
          >
            <Menu size={24} className="text-gray-600" />
          </button>
        </div>
      </div>

      {/* Home View */}
      {currentView === 'home' && (
        <div className="max-w-4xl mx-auto px-4 py-8">
          {/* Mood Check-in */}
          <div className="bg-white rounded-2xl shadow-lg p-6 mb-6">
            <h2 className="text-2xl font-bold text-gray-800 mb-2 flex items-center gap-2">
              <Heart className="text-pink-500" />
              How are you feeling today?
            </h2>
            <p className="text-gray-600 mb-4">Select your current mood to personalize your experience</p>
            
            <div className="grid grid-cols-3 gap-3">
              {moodOptions.map((mood) => (
                <button
                  key={mood.label}
                  onClick={() => handleMoodSelect(mood)}
                  className={`p-4 rounded-xl border-2 transition ${
                    sessionData.mood === mood.label
                      ? 'border-purple-500 bg-purple-50'
                      : 'border-gray-200 hover:border-purple-300'
                  }`}
                >
                  <div className="text-4xl mb-2">{mood.emoji}</div>
                  <div className="text-sm font-medium text-gray-700">{mood.label}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Art Exercises */}
          <div className="bg-white rounded-2xl shadow-lg p-6">
            <h2 className="text-2xl font-bold text-gray-800 mb-2 flex items-center gap-2">
              <Sparkles className="text-purple-500" />
              Mindful Art Exercises
            </h2>
            <p className="text-gray-600 mb-6">Choose an exercise to begin your creative wellness session</p>

            <div className="grid gap-4">
              {artExercises.map((exercise) => (
                <div
                  key={exercise.id}
                  className="border border-gray-200 rounded-xl p-4 hover:shadow-md transition cursor-pointer"
                  onClick={() => startSession(exercise)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <h3 className="font-semibold text-gray-800 mb-1">{exercise.title}</h3>
                      <p className="text-sm text-gray-600">{exercise.description}</p>
                    </div>
                    <div className="bg-purple-100 p-2 rounded-lg">
                      <Palette size={20} className="text-purple-600" />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Chat View */}
      {currentView === 'chat' && (
        <div className="max-w-4xl mx-auto px-4 py-4 h-[calc(100vh-100px)] flex flex-col">
          {/* Exercise Header */}
          <div className="bg-white rounded-xl shadow-sm p-4 mb-4">
            <h3 className="font-semibold text-gray-800">{sessionData.currentExercise?.title}</h3>
            <p className="text-sm text-gray-600">{sessionData.currentExercise?.description}</p>
            {sessionData.mood && (
              <div className="mt-2 inline-block bg-purple-100 px-3 py-1 rounded-full text-sm text-purple-700">
                Mood: {sessionData.mood}
              </div>
            )}
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto bg-white rounded-xl shadow-sm p-4 mb-4 space-y-4">
            {messages.map((msg, idx) => (
              <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div
                  className={`max-w-[80%] p-3 rounded-2xl ${
                    msg.role === 'user'
                      ? 'bg-purple-500 text-white'
                      : 'bg-gray-100 text-gray-800'
                  }`}
                >
                  {msg.content}
                </div>
              </div>
            ))}
            
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-gray-100 p-3 rounded-2xl">
                  <div className="flex gap-1">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                  </div>
                </div>
              </div>
            )}

            {generatedArt && (
              <div className="flex justify-center">
                <div className="bg-white p-4 rounded-xl shadow-lg max-w-md">
                  <p className="text-sm text-gray-600 mb-2 text-center">Generated artwork based on your emotions:</p>
                  <img src={generatedArt} alt="Generated abstract art" className="w-full rounded-lg" />
                  <p className="text-xs text-gray-500 mt-2 text-center">What feelings does this evoke for you?</p>
                </div>
              </div>
            )}
          </div>

          {/* Input */}
          <div className="bg-white rounded-xl shadow-sm p-4 flex gap-2">
            <input
              type="text"
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
              placeholder="Share your thoughts and feelings..."
              className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-purple-500"
              disabled={isLoading}
            />
            <button
              onClick={handleSendMessage}
              disabled={isLoading || !userInput.trim()}
              className="bg-purple-500 text-white p-2 rounded-lg hover:bg-purple-600 disabled:bg-gray-300 transition"
            >
              <Send size={20} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default ArtTherapyCompanion;

