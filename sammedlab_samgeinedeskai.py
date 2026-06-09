# -*- coding: utf-8 -*-
"""
SAMGenieDesk AI - Interactive Dashboard for Kaggle Notebooks

To run this application in a Kaggle or Jupyter notebook,
simply copy and paste the entire content of this script into a single cell and execute it.

This script uses the IPython.display module to render the HTML, CSS, and JavaScript
content directly in the notebook's output cell.
"""

from IPython.display import display, HTML

# The complete HTML, CSS, and JavaScript code is embedded as a multi-line string.
html_content = """
<!DOCTYPE html>
<html lang="ar">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>نظام إدارة تذاكر العملاء - SAMGenieDesk AI</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        body {
            font-family: 'Inter', sans-serif;
            direction: rtl;
            text-align: right;
        }
        .container {
            max-width: 900px;
        }
    </style>
</head>
<body class="bg-gray-100 p-4 sm:p-8">
    <div class="container mx-auto bg-white rounded-xl shadow-lg p-6 sm:p-8">
        <h1 class="text-3xl font-bold text-center mb-6 text-gray-800">SAMGenieDesk AI: لوحة تحليلات تفاعلية لخدمة العملاء</h1>
        
        <div id="main-view">
            <div class="mb-8 p-6 bg-gray-50 rounded-lg border border-gray-200">
                <h2 class="text-2xl font-semibold mb-4 text-gray-700">وصف المشروع</h2>
                <p class="text-gray-600 leading-relaxed">
                    SAMGenieDesk AI هو حل ذكي يعتمد على BigQuery AI لتحويل بيانات تذاكر الدعم الفني غير المهيكلة إلى رؤى قابلة للتنفيذ. يستخدم النظام وظائف الذكاء الاصطناعي التوليدي لتلخيص آلاف المحادثات، وتصنيف الشكاوى بدقة، والتنبؤ باتجاهات العملاء المستقبلية.
                </p>
            </div>

            <div class="mb-8 p-6 bg-blue-50 rounded-lg border-2 border-blue-200">
                <h2 class="text-2xl font-semibold mb-4 text-blue-700">لوحة تحليلات الأداء (وهمية)</h2>
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-4 text-center">
                    <div class="bg-white p-4 rounded-lg shadow-sm border border-gray-200">
                        <p class="text-4xl font-bold text-blue-600">75</p>
                        <p class="text-sm text-gray-500">متوسط زمن الاستجابة (دقيقة)</p>
                    </div>
                    <div class="bg-white p-4 rounded-lg shadow-sm border border-gray-200">
                        <p class="text-4xl font-bold text-blue-600">88%</p>
                        <p class="text-sm text-gray-500">معدل رضا العملاء</p>
                    </div>
                    <div class="bg-white p-4 rounded-lg shadow-sm border border-gray-200">
                        <p class="text-4xl font-bold text-blue-600">تقني</p>
                        <p class="text-sm text-gray-500">أكثر فئة مشاكل شيوعًا</p>
                    </div>
                    <div class="bg-white p-4 rounded-lg shadow-sm border border-gray-200">
                        <p class="text-4xl font-bold text-blue-600">20</p>
                        <p class="text-sm text-gray-500">تذكرة متوقعة غدًا</p>
                    </div>
                </div>
            </div>

            <div class="mb-8 p-6 bg-gray-50 rounded-lg border border-gray-200">
                <h2 class="text-2xl font-semibold mb-4 text-gray-700">إضافة تذكرة جديدة</h2>
                <form id="add-ticket-form" class="space-y-4">
                    <div>
                        <label for="customer_name" class="block text-sm font-medium text-gray-700">اسم العميل</label>
                        <input type="text" id="customer_name" required class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm">
                    </div>
                    <div>
                        <label for="issue_summary" class="block text-sm font-medium text-gray-700">ملخص المشكلة</label>
                        <input type="text" id="issue_summary" required class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm">
                    </div>
                    <div>
                        <label for="issue_date" class="block text-sm font-medium text-gray-700">تاريخ المشكلة</label>
                        <input type="date" id="issue_date" required class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm">
                    </div>
                    <div>
                        <label for="conversation_text" class="block text-sm font-medium text-gray-700">نص المحادثة</label>
                        <textarea id="conversation_text" rows="3" required class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"></textarea>
                    </div>
                    <button type="submit" class="w-full bg-indigo-600 text-white py-2 px-4 rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition duration-150 ease-in-out">
                        إضافة تذكرة
                    </button>
                </form>
            </div>

            <div>
                <h2 class="text-2xl font-semibold mb-4 text-gray-700">قائمة التذاكر الحالية</h2>
                <div id="tickets-container" class="grid gap-4">
                    </div>
            </div>
        </div>

        <div id="detail-view" class="hidden">
            <button id="back-button" class="mb-4 text-blue-600 font-semibold hover:underline">
                <span class="mr-2">←</span> العودة للقائمة الرئيسية
            </button>
            <div id="ticket-detail-card" class="bg-gray-50 p-6 rounded-lg shadow-lg border-2 border-indigo-300">
                </div>
        </div>

        <div class="mt-12 p-6 bg-gray-50 rounded-lg border border-gray-200">
            <h2 class="text-2xl font-semibold mb-4 text-gray-700">خطة التنفيذ: أكواد BigQuery AI</h2>
            <p class="text-gray-600 mb-4">هذه الأكواد تمثل المراحل الرئيسية لمشروع SAMGenieDesk AI. يمكن نسخها ولصقها في بيئة BigQuery لتحليل البيانات.</p>
            
            <h3 class="text-xl font-medium mb-2 text-gray-800">المرحلة الأولى: التلخيص والتصنيف</h3>
            <p class="text-gray-600 mb-4">
                يستخدم هذا الكود وظيفة <code class="bg-gray-200 rounded-md px-1 py-0.5 text-sm">ML.GENERATE_TEXT</code> لتلخيص وتصنيف كل تذكرة دعم فني تلقائيًا، مما يوفر الوقت ويساعد على فهم طبيعة المشاكل بشكل فوري.
            </p>
            <pre class="bg-gray-800 text-white p-4 rounded-md text-sm overflow-x-auto">
<code class="language-sql">
SELECT
  ticket_id,
  customer_name,
  issue_date,
  ML.GENERATE_TEXT(
    MODEL `your-project-id.customer_support_project.summarization_model`,
    (SELECT FORMAT("""
      You are an AI assistant for a customer support team.
      Summarize the following conversation in one sentence and classify it into one of these categories:
      - technical
      - billing
      - shipping
      - general_inquiry
      
      Conversation: "%s"
      
      Example Output:
      Summary: The customer is having a technical issue with their modem.
      Category: technical
      
      Your Output:
      Summary:
      Category:
      """, conversation_text) FROM UNNEST([])),
    options(destination_table = '`your-project-id.customer_support_project.categorized_tickets`')
  ) AS generated_output
FROM
  `your-project-id.customer_support_project.your_csv_table`;
</code>
</pre>

            <h3 class="text-xl font-medium mt-6 mb-2 text-gray-800">المرحلة الثانية: التنبؤ (Forecasting)</h3>
            <p class="text-gray-600 mb-4">
                يستخدم هذا الكود نموذج <code class="bg-gray-200 rounded-md px-1 py-0.5 text-sm">ARIMA_PLUS</code> للتنبؤ بعدد التذاكر المتوقعة في الأيام القادمة، مما يساعد على التخطيط للموارد.
            </p>
            <pre class="bg-gray-800 text-white p-4 rounded-md text-sm overflow-x-auto">
<code class="language-sql">
CREATE OR REPLACE MODEL `your-project-id.customer_support_project.tickets_forecast`
OPTIONS(model_type='ARIMA_PLUS', time_series_timestamp_col='issue_date', time_series_data_col='ticket_count') AS
SELECT
  issue_date,
  COUNT(ticket_id) AS ticket_count
FROM
  `your-project-id.customer_support_project.categorized_tickets`
GROUP BY
  issue_date;

SELECT
  *
FROM
  ML.FORECAST(MODEL `your-project-id.customer_support_project.tickets_forecast`,
    STRUCT(14 AS horizon));
</code>
</pre>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', () => {
            // Mock Data for the ticketing system. In a real app, this would be from an API.
            const initialTickets = [
                {
                    Ticket_id: 1001,
                    customer_name: "أحمد علي",
                    issue_date: "2025-08-15",
                    issue_summary: "عدم تفعيل الخدمة",
                    conversation_text: "مرحباً، حاولت تفعيل خدمة الإنترنت الجديدة لكن لم يتم التفعيل. يظهر لي خطأ في النظام. ما الحل؟",
                    category: "تقني",
                    ai_summary: "المشكلة: العميل يواجه صعوبة في تفعيل خدمة الإنترنت الجديدة.",
                    ai_responses: ["عزيزي العميل، يرجى التأكد من أن جميع الكابلات موصلة بشكل صحيح ثم إعادة تشغيل جهاز المودم.", "أعتذر عن هذه المشكلة. لقد قمنا بإعادة تفعيل الخدمة من جانبنا، يرجى المحاولة مرة أخرى.", "يرجى تزويدنا برقم حسابك لنتمكن من مراجعة حالة التفعيل." ]
                },
                {
                    Ticket_id: 1002,
                    customer_name: "فاطمة سعيد",
                    issue_date: "2025-08-16",
                    issue_summary: "خطأ في الفاتورة",
                    conversation_text: "قمت بمراجعة فاتورتي الأخيرة ووجدت رسومًا إضافية لم أطلبها. يرجى توضيح الأمر أو إزالة الرسوم.",
                    category: "فواتير",
                    ai_summary: "المشكلة: العميل يستفسر عن رسوم إضافية غير معروفة في الفاتورة.",
                    ai_responses: ["عزيزي العميل، لقد تم إزالة الرسوم الإضافية. ستظهر في فاتورتك القادمة.", "يرجى تزويدنا بنسخة من فاتورتك الأخيرة لمراجعة الرسوم.", "هذه الرسوم هي مقابل الاشتراك في خدمة إضافية تم تفعيلها. هل ترغب في إلغائها؟"]
                },
                {
                    Ticket_id: 1003,
                    customer_name: "يوسف عمر",
                    issue_date: "2025-08-17",
                    issue_summary: "مشكلة تقنية",
                    conversation_text: "توقف جهاز المودم عن العمل فجأة، الضوء الأحمر يومض بشكل مستمر. هل يمكن إرسال فني؟",
                    category: "تقني",
                    ai_summary: "المشكلة: جهاز المودم الخاص بالعميل لا يعمل بشكل صحيح.",
                    ai_responses: ["نعتذر عن المشكلة. لقد قمنا بإنشاء تذكرة لإرسال فني إلى موقعك.", "يرجى فصل جهاز المودم من الكهرباء لمدة 30 ثانية ثم إعادة توصيله.","هل جربت إعادة ضبط المصنع للجهاز؟"]
                },
                {
                    Ticket_id: 1004,
                    customer_name: "ليلى حسن",
                    issue_date: "2025-08-18",
                    issue_summary: "استفسار عن شحن",
                    conversation_text: "متى سيتم شحن طلبي؟ رقم الطلب هو 55896. لم أتلق أي تحديثات منذ أسبوع.",
                    category: "شحن",
                    ai_summary: "المشكلة: العميل يستعلم عن حالة شحن طلب لم يتم تحديثه منذ فترة.",
                    ai_responses: ["عزيزي العميل، لقد تم شحن طلبك أمس، ونتوقع وصوله خلال 3 أيام عمل.", "سأقوم بمراجعة قسم الشحن وتزويدك بتحديثات فورية.", "يرجى تتبع شحنتك باستخدام الرابط التالي: [رابط التتبع]"]
                },
                {
                    Ticket_id: 1005,
                    customer_name: "عمر خالد",
                    issue_date: "2025-08-19",
                    issue_summary: "مشكلة في سرعة الإنترنت",
                    conversation_text: "سرعة الإنترنت بطيئة جداً في المساء. لا أستطيع مشاهدة الفيديوهات بجودة عالية. ما السبب؟",
                    category: "تقني",
                    ai_summary: "المشكلة: العميل يشتكي من بطء سرعة الإنترنت في أوقات الذروة.",
                    ai_responses: ["نعتذر عن هذا الإزعاج. قد يكون هناك صيانة في منطقتك. يرجى التحقق من صفحة الصيانة لدينا.", "يرجى التأكد من أن جهاز الراوتر الخاص بك لا يتم استخدامه بشكل مكثف من قبل أجهزة أخرى.", "قم بفحص السرعة عبر هذا الرابط: [رابط فحص السرعة] وأخبرنا بالنتائج."]
                }
            ];

            let ticketsData = [...initialTickets];
            const form = document.getElementById('add-ticket-form');
            const ticketsContainer = document.getElementById('tickets-container');
            const mainView = document.getElementById('main-view');
            const detailView = document.getElementById('detail-view');
            const backButton = document.getElementById('back-button');
            const ticketDetailCard = document.getElementById('ticket-detail-card');
            let nextTicketId = 1006;

            /**
             * Renders a single ticket as an HTML card for the list view.
             * @param {object} ticket - The ticket object to render.
             */
            const renderTicketCard = (ticket) => {
                const ticketCard = document.createElement('div');
                ticketCard.className = 'bg-white p-6 rounded-lg shadow-md border border-gray-200 cursor-pointer hover:bg-gray-50 transition duration-150 ease-in-out';
                ticketCard.dataset.ticketId = ticket.Ticket_id;
                
                ticketCard.innerHTML = `
                    <div class="flex justify-between items-center mb-2">
                        <h3 class="text-xl font-bold text-indigo-600">تذكرة #${ticket.Ticket_id}</h3>
                        <span class="bg-indigo-100 text-indigo-800 text-xs font-semibold px-2 py-1 rounded-full">${ticket.category}</span>
                    </div>
                    <p class="text-lg font-semibold text-gray-800">${ticket.customer_name}</p>
                    <p class="text-gray-600 mt-2 text-sm truncate"><span class="font-medium">ملخص المشكلة:</span> ${ticket.issue_summary}</p>
                    <p class="text-sm text-gray-400 mt-2">${ticket.issue_date}</p>
                `;
                ticketsContainer.appendChild(ticketCard);

                // Add event listener to show detailed view
                ticketCard.addEventListener('click', () => {
                    renderTicketDetail(ticket);
                });
            };

            /**
             * Renders the detailed view for a single ticket.
             * @param {object} ticket - The ticket object to render.
             */
            const renderTicketDetail = (ticket) => {
                mainView.classList.add('hidden');
                detailView.classList.remove('hidden');

                const suggestedResponsesHtml = ticket.ai_responses.map(response => `
                    <div class="bg-white p-4 rounded-md shadow-sm border border-gray-200 mb-2">
                        <p class="text-gray-700 leading-snug">${response}</p>
                        <button class="mt-2 text-sm text-blue-500 hover:underline">إرسال الرد</button>
                    </div>
                `).join('');

                ticketDetailCard.innerHTML = `
                    <div class="flex items-center justify-between mb-4 pb-4 border-b border-gray-200">
                        <div>
                            <h3 class="text-2xl font-bold text-indigo-600">تذكرة #${ticket.Ticket_id}</h3>
                            <p class="text-lg font-semibold text-gray-800">${ticket.customer_name}</p>
                        </div>
                        <span class="bg-indigo-100 text-indigo-800 text-sm font-semibold px-3 py-1 rounded-full">${ticket.category}</span>
                    </div>

                    <div class="mb-6">
                        <p class="text-gray-600 leading-relaxed"><span class="font-bold">ملخص المشكلة:</span> ${ticket.issue_summary}</p>
                        <p class="text-gray-600 leading-relaxed"><span class="font-bold">تاريخ المشكلة:</span> ${ticket.issue_date}</p>
                    </div>
                    
                    <div class="p-4 bg-gray-100 rounded-md mb-6">
                        <h4 class="font-bold text-gray-800 mb-2">نص المحادثة الكامل</h4>
                        <p class="text-gray-700 leading-relaxed">${ticket.conversation_text}</p>
                    </div>

                    <div class="p-4 bg-purple-100 rounded-md mb-6 border-2 border-purple-300">
                        <h4 class="font-bold text-purple-800 mb-2">تحليل الذكاء الاصطناعي</h4>
                        <p class="text-gray-700 leading-relaxed"><span class="font-bold">تلخيص AI:</span> ${ticket.ai_summary}</p>
                    </div>
                    
                    <div class="p-4 bg-green-100 rounded-md border-2 border-green-300">
                        <h4 class="font-bold text-green-800 mb-4">الردود المقترحة من AI</h4>
                        ${suggestedResponsesHtml}
                    </div>
                `;
            };

            /**
             * Renders all tickets from the current data array.
             */
            const renderAllTickets = () => {
                ticketsContainer.innerHTML = '';
                ticketsData.forEach(ticket => renderTicketCard(ticket));
            };

            // Initial render of the default tickets
            renderAllTickets();

            // Handle form submission to add a new ticket
            form.addEventListener('submit', (event) => {
                event.preventDefault();

                // In a real application, this data would be sent to the BigQuery AI API for analysis.
                // Here, we use placeholder data for demonstration purposes.
                const newTicket = {
                    Ticket_id: nextTicketId++,
                    customer_name: document.getElementById('customer_name').value,
                    issue_summary: document.getElementById('issue_summary').value,
                    issue_date: document.getElementById('issue_date').value,
                    conversation_text: document.getElementById('conversation_text').value,
                    category: "استفسار عام", 
                    ai_summary: "ملخص وهمي: هذا ملخص تم إنشاؤه للتوضيح.", 
                    ai_responses: ["رد آلي: شكراً لتواصلك. سنقوم بمراجعة طلبك والرد عليك قريباً.", "رد آلي: يرجى تزويدنا بمعلومات إضافية لمساعدتك بشكل أفضل." ] 
                };

                ticketsData.push(newTicket);
                form.reset();
                renderAllTickets();
            });

            // Handle the back button click
            backButton.addEventListener('click', () => {
                detailView.classList.add('hidden');
                mainView.classList.remove('hidden');
            });
        });
    </script>
</body>
</html>
"""

# Display the HTML content in the notebook's output cell
display(HTML(html_content))







