# auth.py
import streamlit as st

# ==========================
# 用户表
# ==========================
USERS = {
    "15908130915": {
        "password": "zrty600030",
        "role": "viewer",
        "name": "weikang",
        "email": "tanwk@chinassic.com",
    },
    "13808003372": {
        "password": "zrty600030",
        "role": "viewer",
        "name": "hezong",
        "email": "hey@chinassic.com",
    },
    "18608088518": {
        "password": "zrty600030",
        "role": "viewer",
        "name": "rui",
        "email": "hezj@chinassic.com",
    },
    "13688128302": {
        "password": "zrty600030",
        "role": "viewer",
        "name": "ziyi",
        "email": "yangzy@chinassic.com",
    },
    "18684094521": {
        "password": "zrty600030",
        "role": "viewer",
        "name": "xiaokun",
        "email": "yexk@chinassic.com",
    },
    "18215588237": {
        "password": "zrty600030",
        "role": "viewer",
        "name": "kunshan",
        "email": "yangks@chinassic.com",
    },
    "13438216458": {
        "password": "zrty600030",
        "role": "viewer",
        "name": "miao",
        "email": "zhengm@chinassic.com",
    },
    "13229514572": {
        "password": "zrty600030",
        "role": "viewer",
        "name": "jixin",
        "email": "yinjx@chinassic.com",
    },
    "18884558596": {
        "password": "zrty600030",
        "role": "viewer",
        "name": "zhiwei",
        "email": "xianzw@chinassic.com",
    },
    "admin": {
        "password": "a12345679",
        "role": "admin",
        "name": "zijie",
        "email": "henryovo62@gmail.com",
    },
}

# ==========================
# 公安备案图片
# ==========================
_BEIAN_IMG_BASE64_RAW = """
iVBORw0KGgoAAAANSUhEUgAAACQAAAAoCAYAAACWwljjAAAFQklEQVRYw+3Wa1BUdRjH8SOpMeg4
WhZGpDIxiaaTeUFgWrxE4AVRQJGlRRAVIV1JkbgMgQLi5AVBQSVLSp0xlEAUKBEEFZCrCstll8UV
2AV2YbmoGCrYv31+R95UL5pmmtamZ+bz6rz5nvOc/5zDcX9jGLs/iTxuyvIlWYkRFeTHA2HVRFtz
fhthTG5KuH96/vUgNlC4mMgyw1NJit/aAXLKazYje9xtIMZ/OZz50gW+9hcNkvoLEemEPbnrSP47
QYwxQ5Ifv54RqzcXwFFvSyjaOhfavN8F7Y5ZcC/HH9JOB4LNa9Zw5YA76OZV8vIGMdZtSp7cDrtO
nOavYiQhTAiPwi1AMtIQaqyngsxpBtw2GAGDKfaQmpUAa6xc4Vfp4UtEdzAMycsT9JQ1Tyctl/2e
EkuTlYysF/rCUNxMqDEzgTqzSXBnpgnIHCzgjvEEuD52DLBr3rA1MAaWmNtB582wdtIljZ9G9D+I
PU6aTxIPBjHCcXvg3CEh9K2fDLWvjIH6D6fwTIyheuwEqLUyhzLOALq8pkN+bgRw3HY4FBsMzxoj
ZxP9DequLjAlQwVrbpIjhyIY4UYGQ/buhdBqPxlk3Gion2IMDQIz3kJe/ZS34I7uHkmD7VSQVgYD
NyIAwsNCgfXGXoOBPjP9DKrOCAogA2etGTmTHAMcFwFZye7wS5QlVHGjoEw4A2qPCUBZ6AzNcQ5Q
/YYRdO+YB1U3dsDwypLio4FJ3ECryIzWz6Cm3NgTRHN8HiPF6eHAGSbAdh8feFZkB7krzaHE9h2o
85sDsiAbkIsXQMN+e2CtGyF0kzdwXCgU5++D/ouLQFV4OEU/g2Q/iNuIPNaKkQflAWBqexxGjhLD
VUcL6IwSQN3SGVChe6FJg9dckCx6D1QBliDZLIAxo7eA8eyv4KE0BJqTrHkZvnL9DJKn+Twmt0Ns
GGHZy2Dn3kQYfsQ53Hh4/r4RNGz8AIpdzKEuaAF0RC2E57MmQgE3ATjuM/CPiANW7AqSfQJQ5vk3
62eQKmd3JrmXsoSRocpNIMnbB9zbceDIWUPmuHFQNMkISqa9DpUvNK6YDpW2s8DfwBK48WFQnhMC
gzUBoLy0BrRVe5P0NWjPLdKUsJiR1tR1wGp8IeZwMgx/SrgRvjxuAziNcwLvyathLOcJHLflhRDY
GRYFrNET2rJ5yvPLoas0tOj/oL8UpC4JHyTSU+6MNCS4gvKoAB5WiKG+MAQSg0WwLXQ/ZJ3xhao0
FxB5hYCbUwAEfhEF3Td8QP2dAOQnPwFlxgrolUVq9TPoaX+ZB2nLc2Gk6awj1MU78HZZwJMid2By
b550JQwVO0NfxlJgdz14vWKeRAiK6DlQF28PLZdcoLNcBIO92bb6GTQ8Q/13RURT6tlH2gvXMlIT
LYD6uI+gp2ozdF0VQXumM6ivCqGvahM8kPiDItkeGo8tB025GFQ3xFrSr06zI3/4yde7oN7m0sWk
5eKWDqK5JWJQvAHac9ygq3Adr9gTNNc3QG85rzPfHe5/7wDtPwuhp/Zz6CjyhaZzwi6ivfetHdH/
oP77+3PJQOsuRnqkQdCa4wWqyx6gyecpL64GTaEX7ycXUJz4GJp1B4O0X/Hg0Xp1tFV+8Ei1k6c5
coHofxBrrzQinbKYo0SVJ+wn6iurGHlY5gY911aDJnMFaHXXiDp9GQyvtKfUA9QFTtBZ7gPdit0t
pFd9OpwwFmlA9D/o9yNLDpxIKmI8PMnNSNtviCLVpYTITzrXEGWaq4qos0WgOPdpCenIF+eRrurj
B4k0PXopYZG6gMg/D/gNBUxhAbSAmKMAAAAASUVORK5CYII=
"""

_BEIAN_IMG_BASE64 = "".join(_BEIAN_IMG_BASE64_RAW.split())


def render_beian_footer():
    st.markdown(
        f"""
        <style>
        .beian-footer {{
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            text-align: center;
            font-size: 12px;
            color: #666;
            padding: 6px 0;
            background-color: rgba(255,255,255,0.95);
            z-index: 999;
        }}
        .beian-footer img {{
            height: 14px;
            vertical-align: middle;
            margin-right: 4px;
        }}
        </style>

        <div class="beian-footer">
            <img src="data:image/png;base64,{_BEIAN_IMG_BASE64}" />
            <a href="https://beian.mps.gov.cn/#/query/webSearch?code=54010002000237"
               target="_blank"
               rel="noreferrer">
               藏公网安备54010002000237号
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================
# Auth 主函数
# ==========================
def require_login():
    if st.session_state.get("logged_in"):
        return True

    st.title("📈 中睿投研")

    username = st.text_input("用户名")
    password = st.text_input("密码", type="password")

    if st.button("登录"):
        user = USERS.get(username)

        if not user:
            st.error("用户不存在")
            render_beian_footer()
            return False

        if password != user["password"]:
            st.error("密码错误")
            render_beian_footer()
            return False

        # 登录成功
        st.session_state.logged_in = True
        st.session_state.username = username
        st.session_state.role = user["role"]
        st.session_state.name = user["name"]
        st.session_state.email = user["email"]

        st.rerun()

    render_beian_footer()
    return False