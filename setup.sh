mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"rahul.kumeriya@outlook.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" >> ~/.streamlit/config.toml

echo "\
[theme]\n\
primaryColor=\"#2214c7\"
backgroundColor=\"#ffffff\"
secondaryBackgroundColor=\"#e8eef9\"
textColor=\"#000000\"\n\
font=\"sans serif\"\n
" >> ~/.streamlit/config.toml