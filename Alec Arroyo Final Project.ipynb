{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "02a23342",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Alec Arroyo\n",
    "#Music Genre Classification\n",
    "\n",
    "import pandas as pd\n",
    "import nltk\n",
    "from pandasql import sqldf\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from nltk.corpus import stopwords\n",
    "from nltk.stem import PorterStemmer\n",
    "import string\n",
    "from sklearn.metrics import confusion_matrix\n",
    "from imblearn.over_sampling import RandomOverSampler\n",
    "from collections import Counter"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "6f8b092f",
   "metadata": {},
   "outputs": [],
   "source": [
    "fileartist = pd.read_csv('/Users/alec_arroyo/Documents/Sryacuse Data Science Courses/Text Mining/artists-data.csv')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "d27ed5f7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Artist</th>\n",
       "      <th>Songs</th>\n",
       "      <th>Popularity</th>\n",
       "      <th>Link</th>\n",
       "      <th>Genre</th>\n",
       "      <th>Genres</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>10000 Maniacs</td>\n",
       "      <td>110</td>\n",
       "      <td>0.3</td>\n",
       "      <td>/10000-maniacs/</td>\n",
       "      <td>Rock</td>\n",
       "      <td>Rock; Pop; Electronica; Dance; J-Pop/J-Rock; G...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>12 Stones</td>\n",
       "      <td>75</td>\n",
       "      <td>0.3</td>\n",
       "      <td>/12-stones/</td>\n",
       "      <td>Rock</td>\n",
       "      <td>Rock; Gospel/Religioso; Hard Rock; Grunge; Roc...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>311</td>\n",
       "      <td>196</td>\n",
       "      <td>0.5</td>\n",
       "      <td>/311/</td>\n",
       "      <td>Rock</td>\n",
       "      <td>Rock; Surf Music; Reggae; Ska; Pop/Rock; Rock ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4 Non Blondes</td>\n",
       "      <td>15</td>\n",
       "      <td>7.5</td>\n",
       "      <td>/4-non-blondes/</td>\n",
       "      <td>Rock</td>\n",
       "      <td>Rock; Pop/Rock; Rock Alternativo; Grunge; Blue...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>A Cruz Está Vazia</td>\n",
       "      <td>13</td>\n",
       "      <td>0.0</td>\n",
       "      <td>/a-cruz-esta-vazia/</td>\n",
       "      <td>Rock</td>\n",
       "      <td>Rock</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>Aborto Elétrico</td>\n",
       "      <td>36</td>\n",
       "      <td>0.1</td>\n",
       "      <td>/aborto-eletrico/</td>\n",
       "      <td>Rock</td>\n",
       "      <td>Rock; Punk Rock; Pós-Punk; Post-Rock</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>Abril</td>\n",
       "      <td>36</td>\n",
       "      <td>0.1</td>\n",
       "      <td>/abril/</td>\n",
       "      <td>Rock</td>\n",
       "      <td>Rock; Emocore; Hardcore; Pop/Rock; Rock Altern...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>Abuse</td>\n",
       "      <td>13</td>\n",
       "      <td>0.0</td>\n",
       "      <td>/abuse/</td>\n",
       "      <td>Rock</td>\n",
       "      <td>Rock; Hardcore</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>AC/DC</td>\n",
       "      <td>192</td>\n",
       "      <td>10.8</td>\n",
       "      <td>/ac-dc/</td>\n",
       "      <td>Rock</td>\n",
       "      <td>Rock; Heavy Metal; Classic Rock; Hard Rock; Cl...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>ACEIA</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>/aceia/</td>\n",
       "      <td>Rock</td>\n",
       "      <td>Rock</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>Acid Tree</td>\n",
       "      <td>5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>/acid-tree/</td>\n",
       "      <td>Rock</td>\n",
       "      <td>Rock; Heavy Metal; Metal</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>Adam Lambert</td>\n",
       "      <td>110</td>\n",
       "      <td>1.4</td>\n",
       "      <td>/adam-lambert/</td>\n",
       "      <td>Pop</td>\n",
       "      <td>Pop; Pop/Rock; Rock; Romântico; Dance; Electro...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>Adrian Suirady</td>\n",
       "      <td>7</td>\n",
       "      <td>0.0</td>\n",
       "      <td>/adrian-suirady/</td>\n",
       "      <td>Rock</td>\n",
       "      <td>Rock; Gótico</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>Aerosmith</td>\n",
       "      <td>249</td>\n",
       "      <td>16.5</td>\n",
       "      <td>/aerosmith/</td>\n",
       "      <td>Rock</td>\n",
       "      <td>Rock; Hard Rock; Heavy Metal; Romântico; Pop/R...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>Aliados</td>\n",
       "      <td>75</td>\n",
       "      <td>0.8</td>\n",
       "      <td>/aliados/</td>\n",
       "      <td>Rock</td>\n",
       "      <td>Rock; Pop/Rock; Rock Alternativo; Surf Music; ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>Alice Cooper</td>\n",
       "      <td>310</td>\n",
       "      <td>1.2</td>\n",
       "      <td>/alice-cooper/</td>\n",
       "      <td>Rock</td>\n",
       "      <td>Rock; Hard Rock; Heavy Metal; Punk Rock; Class...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>Alter Bridge</td>\n",
       "      <td>74</td>\n",
       "      <td>1.4</td>\n",
       "      <td>/alter-bridge/</td>\n",
       "      <td>Rock</td>\n",
       "      <td>Rock; Hard Rock; Rock Alternativo; Heavy Metal...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>Amy Lee</td>\n",
       "      <td>33</td>\n",
       "      <td>0.5</td>\n",
       "      <td>/amy-lee/</td>\n",
       "      <td>Rock</td>\n",
       "      <td>Rock; Gótico; Hard Rock; Rock Alternativo; Hea...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>Anberlin</td>\n",
       "      <td>98</td>\n",
       "      <td>0.1</td>\n",
       "      <td>/anberlin/</td>\n",
       "      <td>Rock</td>\n",
       "      <td>Rock; Rock Alternativo; Hardcore; Emocore; Gos...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>Andi Deris</td>\n",
       "      <td>44</td>\n",
       "      <td>0.0</td>\n",
       "      <td>/andi-deris/</td>\n",
       "      <td>Rock</td>\n",
       "      <td>Rock; Hard Rock; Heavy Metal</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               Artist  Songs  Popularity                 Link Genre  \\\n",
       "0       10000 Maniacs    110         0.3      /10000-maniacs/  Rock   \n",
       "1           12 Stones     75         0.3          /12-stones/  Rock   \n",
       "2                 311    196         0.5                /311/  Rock   \n",
       "3       4 Non Blondes     15         7.5      /4-non-blondes/  Rock   \n",
       "4   A Cruz Está Vazia     13         0.0  /a-cruz-esta-vazia/  Rock   \n",
       "5     Aborto Elétrico     36         0.1    /aborto-eletrico/  Rock   \n",
       "6               Abril     36         0.1              /abril/  Rock   \n",
       "7               Abuse     13         0.0              /abuse/  Rock   \n",
       "8               AC/DC    192        10.8              /ac-dc/  Rock   \n",
       "9               ACEIA      0         0.0              /aceia/  Rock   \n",
       "10          Acid Tree      5         0.0          /acid-tree/  Rock   \n",
       "11       Adam Lambert    110         1.4       /adam-lambert/   Pop   \n",
       "12     Adrian Suirady      7         0.0     /adrian-suirady/  Rock   \n",
       "13          Aerosmith    249        16.5          /aerosmith/  Rock   \n",
       "14            Aliados     75         0.8            /aliados/  Rock   \n",
       "15       Alice Cooper    310         1.2       /alice-cooper/  Rock   \n",
       "16       Alter Bridge     74         1.4       /alter-bridge/  Rock   \n",
       "17            Amy Lee     33         0.5            /amy-lee/  Rock   \n",
       "18           Anberlin     98         0.1           /anberlin/  Rock   \n",
       "19         Andi Deris     44         0.0         /andi-deris/  Rock   \n",
       "\n",
       "                                               Genres  \n",
       "0   Rock; Pop; Electronica; Dance; J-Pop/J-Rock; G...  \n",
       "1   Rock; Gospel/Religioso; Hard Rock; Grunge; Roc...  \n",
       "2   Rock; Surf Music; Reggae; Ska; Pop/Rock; Rock ...  \n",
       "3   Rock; Pop/Rock; Rock Alternativo; Grunge; Blue...  \n",
       "4                                                Rock  \n",
       "5                Rock; Punk Rock; Pós-Punk; Post-Rock  \n",
       "6   Rock; Emocore; Hardcore; Pop/Rock; Rock Altern...  \n",
       "7                                      Rock; Hardcore  \n",
       "8   Rock; Heavy Metal; Classic Rock; Hard Rock; Cl...  \n",
       "9                                                Rock  \n",
       "10                           Rock; Heavy Metal; Metal  \n",
       "11  Pop; Pop/Rock; Rock; Romântico; Dance; Electro...  \n",
       "12                                       Rock; Gótico  \n",
       "13  Rock; Hard Rock; Heavy Metal; Romântico; Pop/R...  \n",
       "14  Rock; Pop/Rock; Rock Alternativo; Surf Music; ...  \n",
       "15  Rock; Hard Rock; Heavy Metal; Punk Rock; Class...  \n",
       "16  Rock; Hard Rock; Rock Alternativo; Heavy Metal...  \n",
       "17  Rock; Gótico; Hard Rock; Rock Alternativo; Hea...  \n",
       "18  Rock; Rock Alternativo; Hardcore; Emocore; Gos...  \n",
       "19                       Rock; Hard Rock; Heavy Metal  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "fileartist[:20]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "bbda4f51",
   "metadata": {},
   "outputs": [],
   "source": [
    "filelyrics = pd.read_csv('/Users/alec_arroyo/Documents/Sryacuse Data Science Courses/Text Mining/lyrics-data.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "edb1776e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ALink</th>\n",
       "      <th>SName</th>\n",
       "      <th>SLink</th>\n",
       "      <th>Lyric</th>\n",
       "      <th>Idiom</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>/10000-maniacs/</td>\n",
       "      <td>More Than This</td>\n",
       "      <td>/10000-maniacs/more-than-this.html</td>\n",
       "      <td>I could feel at the time. There was no way of ...</td>\n",
       "      <td>ENGLISH</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>/10000-maniacs/</td>\n",
       "      <td>Because The Night</td>\n",
       "      <td>/10000-maniacs/because-the-night.html</td>\n",
       "      <td>Take me now, baby, here as I am. Hold me close...</td>\n",
       "      <td>ENGLISH</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>/10000-maniacs/</td>\n",
       "      <td>These Are Days</td>\n",
       "      <td>/10000-maniacs/these-are-days.html</td>\n",
       "      <td>These are. These are days you'll remember. Nev...</td>\n",
       "      <td>ENGLISH</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>/10000-maniacs/</td>\n",
       "      <td>A Campfire Song</td>\n",
       "      <td>/10000-maniacs/a-campfire-song.html</td>\n",
       "      <td>A lie to say, \"O my mountain has coal veins an...</td>\n",
       "      <td>ENGLISH</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>/10000-maniacs/</td>\n",
       "      <td>Everyday Is Like Sunday</td>\n",
       "      <td>/10000-maniacs/everyday-is-like-sunday.html</td>\n",
       "      <td>Trudging slowly over wet sand. Back to the ben...</td>\n",
       "      <td>ENGLISH</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>/10000-maniacs/</td>\n",
       "      <td>Don't Talk</td>\n",
       "      <td>/10000-maniacs/dont-talk.html</td>\n",
       "      <td>Don't talk, I will listen. Don't talk, you kee...</td>\n",
       "      <td>ENGLISH</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>/10000-maniacs/</td>\n",
       "      <td>Across The Fields</td>\n",
       "      <td>/10000-maniacs/across-the-fields.html</td>\n",
       "      <td>Well they left then in the morning, a hundred ...</td>\n",
       "      <td>ENGLISH</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>/10000-maniacs/</td>\n",
       "      <td>Planned Obsolescence</td>\n",
       "      <td>/10000-maniacs/planned-obsolescence.html</td>\n",
       "      <td>[ music: Dennis Drew/lyric: Natalie Merchant ]...</td>\n",
       "      <td>ENGLISH</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>/10000-maniacs/</td>\n",
       "      <td>Rainy Day</td>\n",
       "      <td>/10000-maniacs/rainy-day.html</td>\n",
       "      <td>On bended kneeI've looked through every window...</td>\n",
       "      <td>ENGLISH</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>/10000-maniacs/</td>\n",
       "      <td>Anthem For Doomed Youth</td>\n",
       "      <td>/10000-maniacs/anthem-for-doomed-youth.html</td>\n",
       "      <td>For whom do the bells toll. When sentenced to ...</td>\n",
       "      <td>ENGLISH</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>/10000-maniacs/</td>\n",
       "      <td>All That Never Happens</td>\n",
       "      <td>/10000-maniacs/all-that-never-happens.html</td>\n",
       "      <td>She walks alone on the brick lane,. the breeze...</td>\n",
       "      <td>ENGLISH</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>/10000-maniacs/</td>\n",
       "      <td>Back O' The Moon</td>\n",
       "      <td>/10000-maniacs/back-o-the-moon.html</td>\n",
       "      <td>Jenny. Jenny you don't know the nights I hide....</td>\n",
       "      <td>ENGLISH</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>/10000-maniacs/</td>\n",
       "      <td>A Room For Everything</td>\n",
       "      <td>/10000-maniacs/a-room-for-everything.html</td>\n",
       "      <td>You were looking away from me, western skies c...</td>\n",
       "      <td>ENGLISH</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>/10000-maniacs/</td>\n",
       "      <td>Like The Weather</td>\n",
       "      <td>/10000-maniacs/like-the-weather.html</td>\n",
       "      <td>The color of the sky as far as I can see is co...</td>\n",
       "      <td>ENGLISH</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>/10000-maniacs/</td>\n",
       "      <td>Eat For Two</td>\n",
       "      <td>/10000-maniacs/eat-for-two.html</td>\n",
       "      <td>Oh,. Baby blankets and. Baby shoes,. Baby slip...</td>\n",
       "      <td>ENGLISH</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>/10000-maniacs/</td>\n",
       "      <td>Maddox Table</td>\n",
       "      <td>/10000-maniacs/maddox-table.html</td>\n",
       "      <td>the legs of Maddox kitchen tables. my whole li...</td>\n",
       "      <td>ENGLISH</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>/10000-maniacs/</td>\n",
       "      <td>Can't Ignore The Train</td>\n",
       "      <td>/10000-maniacs/cant-ignore-the-train.html</td>\n",
       "      <td>Steep is the water tower. painted off blue to ...</td>\n",
       "      <td>ENGLISH</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>/10000-maniacs/</td>\n",
       "      <td>To Sir, With Love</td>\n",
       "      <td>/10000-maniacs/to-sir-with-love.html</td>\n",
       "      <td>[original version by Lulu]. Those schoolgirl d...</td>\n",
       "      <td>ENGLISH</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>/10000-maniacs/</td>\n",
       "      <td>Stockton Gala Days</td>\n",
       "      <td>/10000-maniacs/stockton-gala-days.html</td>\n",
       "      <td>That summer fields grew high with foxglove sta...</td>\n",
       "      <td>ENGLISH</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>/10000-maniacs/</td>\n",
       "      <td>Poison In The Well</td>\n",
       "      <td>/10000-maniacs/poison-in-the-well.html</td>\n",
       "      <td>[ music: Dennis Drew/words: Natalie Merchant ]...</td>\n",
       "      <td>ENGLISH</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              ALink                    SName  \\\n",
       "0   /10000-maniacs/           More Than This   \n",
       "1   /10000-maniacs/        Because The Night   \n",
       "2   /10000-maniacs/           These Are Days   \n",
       "3   /10000-maniacs/          A Campfire Song   \n",
       "4   /10000-maniacs/  Everyday Is Like Sunday   \n",
       "5   /10000-maniacs/               Don't Talk   \n",
       "6   /10000-maniacs/        Across The Fields   \n",
       "7   /10000-maniacs/     Planned Obsolescence   \n",
       "8   /10000-maniacs/                Rainy Day   \n",
       "9   /10000-maniacs/  Anthem For Doomed Youth   \n",
       "10  /10000-maniacs/   All That Never Happens   \n",
       "11  /10000-maniacs/         Back O' The Moon   \n",
       "12  /10000-maniacs/    A Room For Everything   \n",
       "13  /10000-maniacs/         Like The Weather   \n",
       "14  /10000-maniacs/              Eat For Two   \n",
       "15  /10000-maniacs/             Maddox Table   \n",
       "16  /10000-maniacs/   Can't Ignore The Train   \n",
       "17  /10000-maniacs/        To Sir, With Love   \n",
       "18  /10000-maniacs/       Stockton Gala Days   \n",
       "19  /10000-maniacs/       Poison In The Well   \n",
       "\n",
       "                                          SLink  \\\n",
       "0            /10000-maniacs/more-than-this.html   \n",
       "1         /10000-maniacs/because-the-night.html   \n",
       "2            /10000-maniacs/these-are-days.html   \n",
       "3           /10000-maniacs/a-campfire-song.html   \n",
       "4   /10000-maniacs/everyday-is-like-sunday.html   \n",
       "5                 /10000-maniacs/dont-talk.html   \n",
       "6         /10000-maniacs/across-the-fields.html   \n",
       "7      /10000-maniacs/planned-obsolescence.html   \n",
       "8                 /10000-maniacs/rainy-day.html   \n",
       "9   /10000-maniacs/anthem-for-doomed-youth.html   \n",
       "10   /10000-maniacs/all-that-never-happens.html   \n",
       "11          /10000-maniacs/back-o-the-moon.html   \n",
       "12    /10000-maniacs/a-room-for-everything.html   \n",
       "13         /10000-maniacs/like-the-weather.html   \n",
       "14              /10000-maniacs/eat-for-two.html   \n",
       "15             /10000-maniacs/maddox-table.html   \n",
       "16    /10000-maniacs/cant-ignore-the-train.html   \n",
       "17         /10000-maniacs/to-sir-with-love.html   \n",
       "18       /10000-maniacs/stockton-gala-days.html   \n",
       "19       /10000-maniacs/poison-in-the-well.html   \n",
       "\n",
       "                                                Lyric    Idiom  \n",
       "0   I could feel at the time. There was no way of ...  ENGLISH  \n",
       "1   Take me now, baby, here as I am. Hold me close...  ENGLISH  \n",
       "2   These are. These are days you'll remember. Nev...  ENGLISH  \n",
       "3   A lie to say, \"O my mountain has coal veins an...  ENGLISH  \n",
       "4   Trudging slowly over wet sand. Back to the ben...  ENGLISH  \n",
       "5   Don't talk, I will listen. Don't talk, you kee...  ENGLISH  \n",
       "6   Well they left then in the morning, a hundred ...  ENGLISH  \n",
       "7   [ music: Dennis Drew/lyric: Natalie Merchant ]...  ENGLISH  \n",
       "8   On bended kneeI've looked through every window...  ENGLISH  \n",
       "9   For whom do the bells toll. When sentenced to ...  ENGLISH  \n",
       "10  She walks alone on the brick lane,. the breeze...  ENGLISH  \n",
       "11  Jenny. Jenny you don't know the nights I hide....  ENGLISH  \n",
       "12  You were looking away from me, western skies c...  ENGLISH  \n",
       "13  The color of the sky as far as I can see is co...  ENGLISH  \n",
       "14  Oh,. Baby blankets and. Baby shoes,. Baby slip...  ENGLISH  \n",
       "15  the legs of Maddox kitchen tables. my whole li...  ENGLISH  \n",
       "16  Steep is the water tower. painted off blue to ...  ENGLISH  \n",
       "17  [original version by Lulu]. Those schoolgirl d...  ENGLISH  \n",
       "18  That summer fields grew high with foxglove sta...  ENGLISH  \n",
       "19  [ music: Dennis Drew/words: Natalie Merchant ]...  ENGLISH  "
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "filelyrics[:20]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "0eb08bed",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "209522"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Merge into 1 DataFrame\n",
    "\n",
    "len(filelyrics)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "8609e5bf",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "3242"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(fileartist)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "a7bdf7a4",
   "metadata": {},
   "outputs": [],
   "source": [
    "#query = \"\"\"\n",
    "#SELECT Genre, COUNT(*) as Total_Count\n",
    "#FROM fileartist\n",
    "#GROUP BY Genre\n",
    "#ORDER BY Total_Count asc\n",
    "#\"\"\"\n",
    "\n",
    "query1 = sqldf('SELECT * FROM fileartist WHERE Genre = \"Rock\"')\n",
    "query2 = sqldf('SELECT * FROM fileartist WHERE Genre = \"Funk Carioca\"')\n",
    "query3 = sqldf('SELECT * FROM fileartist WHERE Genre = \"Hip Hop\"')\n",
    "query4 = sqldf('SELECT * FROM fileartist WHERE Genre = \"Sertanejo\"')\n",
    "query5 = sqldf('SELECT * FROM fileartist WHERE Genre = \"Pop\"')\n",
    "query6 = sqldf('SELECT * FROM fileartist WHERE Genre = \"Samba\"')\n",
    "\n",
    "totalcounts = sqldf('SELECT Genre, COUNT(*) as Total_Count FROM fileartist GROUP BY Genre ORDER BY Total_Count asc')\n",
    "\n",
    "#new = sqldf.run(query)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "35ad10d3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Genre</th>\n",
       "      <th>Total_Count</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Samba</td>\n",
       "      <td>193</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Funk Carioca</td>\n",
       "      <td>302</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Hip Hop</td>\n",
       "      <td>537</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Sertanejo</td>\n",
       "      <td>617</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Pop</td>\n",
       "      <td>796</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>Rock</td>\n",
       "      <td>797</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          Genre  Total_Count\n",
       "0         Samba          193\n",
       "1  Funk Carioca          302\n",
       "2       Hip Hop          537\n",
       "3     Sertanejo          617\n",
       "4           Pop          796\n",
       "5          Rock          797"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#How many artists for each genre of music\n",
    "totalcounts"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "9bda3ad1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<BarContainer object of 6 artists>"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#How many artists for each genre of music\n",
    "data = {'Rock':len(query1), 'Funk Carioca':len(query2), 'Hip Hop':len(query3), 'Sertanejo':len(query4), 'Pop':len(query5), 'Samba':len(query6)}\n",
    "names = list(data.keys())\n",
    "values = list(data.values())\n",
    "plt.title(\"Num of Artists Per Genre\")\n",
    "plt.bar(names, values)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "2d2340cc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "3242"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(fileartist['Genre'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "4395d799",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "209522"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(filelyrics)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "d0ee7c07",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Artist</th>\n",
       "      <th>Songs</th>\n",
       "      <th>Popularity</th>\n",
       "      <th>SName</th>\n",
       "      <th>Lyric</th>\n",
       "      <th>Genre</th>\n",
       "      <th>Idiom</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Luan Santana</td>\n",
       "      <td>187</td>\n",
       "      <td>17.2</td>\n",
       "      <td>\"A\"</td>\n",
       "      <td>Tá em dúvida. Não sabe se é normal gostar de d...</td>\n",
       "      <td>Sertanejo</td>\n",
       "      <td>PORTUGUESE</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Mutantes</td>\n",
       "      <td>123</td>\n",
       "      <td>1.0</td>\n",
       "      <td>\"A\" e o \"Z\"</td>\n",
       "      <td>Eu sou o começo. Sou o Fim. Sou o A e o Z. Eu ...</td>\n",
       "      <td>Rock</td>\n",
       "      <td>PORTUGUESE</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Foxy Brown</td>\n",
       "      <td>74</td>\n",
       "      <td>0.0</td>\n",
       "      <td>\"Oh Yeah\" By Foxy Brown</td>\n",
       "      <td>[Verse One]. I'm the most critically acclaimed...</td>\n",
       "      <td>Hip Hop</td>\n",
       "      <td>ENGLISH</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Barbie Kue</td>\n",
       "      <td>8</td>\n",
       "      <td>0.0</td>\n",
       "      <td>\"Sei Lá...\"</td>\n",
       "      <td>Se um dia olhar pro lado. Eu não vou mais esta...</td>\n",
       "      <td>Rock</td>\n",
       "      <td>PORTUGUESE</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Tiziano Ferro</td>\n",
       "      <td>160</td>\n",
       "      <td>2.1</td>\n",
       "      <td>\"Solo\" e' Solo Una Parola</td>\n",
       "      <td>Il cuore è andato in guerra ma la vita. non l'...</td>\n",
       "      <td>Pop</td>\n",
       "      <td>ITALIAN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>161284</th>\n",
       "      <td>Ariana Grande</td>\n",
       "      <td>155</td>\n",
       "      <td>246.8</td>\n",
       "      <td>​Goodnight n Go</td>\n",
       "      <td>Tell me why you gotta look at me that way. You...</td>\n",
       "      <td>Pop</td>\n",
       "      <td>ENGLISH</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>161285</th>\n",
       "      <td>Girls' Generation</td>\n",
       "      <td>248</td>\n",
       "      <td>0.4</td>\n",
       "      <td>쉼표 (Fermata)</td>\n",
       "      <td>Maeumi swineun dosi. eopseul georan geol ara. ...</td>\n",
       "      <td>Pop</td>\n",
       "      <td>None</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>161286</th>\n",
       "      <td>BigBang</td>\n",
       "      <td>175</td>\n",
       "      <td>1.0</td>\n",
       "      <td>착한 사람 (a Good Man)</td>\n",
       "      <td>Chakhansaramieosseum angeuraetjyo. Geureona na...</td>\n",
       "      <td>Hip Hop</td>\n",
       "      <td>None</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>161287</th>\n",
       "      <td>BigBang</td>\n",
       "      <td>175</td>\n",
       "      <td>1.0</td>\n",
       "      <td>천국 (heaven)</td>\n",
       "      <td>saranghae nan neol gieokhae. HEAVEN. SING IT T...</td>\n",
       "      <td>Hip Hop</td>\n",
       "      <td>None</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>161288</th>\n",
       "      <td>BigBang</td>\n",
       "      <td>175</td>\n",
       "      <td>1.0</td>\n",
       "      <td>하루 하루 (haru Haru)</td>\n",
       "      <td>YEAH. FINALLY I REALIZED. THAT I’M NOTHING WIT...</td>\n",
       "      <td>Hip Hop</td>\n",
       "      <td>None</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>161289 rows × 7 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                   Artist  Songs  Popularity                      SName  \\\n",
       "0            Luan Santana    187        17.2                        \"A\"   \n",
       "1                Mutantes    123         1.0                \"A\" e o \"Z\"   \n",
       "2              Foxy Brown     74         0.0    \"Oh Yeah\" By Foxy Brown   \n",
       "3              Barbie Kue      8         0.0                \"Sei Lá...\"   \n",
       "4           Tiziano Ferro    160         2.1  \"Solo\" e' Solo Una Parola   \n",
       "...                   ...    ...         ...                        ...   \n",
       "161284      Ariana Grande    155       246.8            ​Goodnight n Go   \n",
       "161285  Girls' Generation    248         0.4               쉼표 (Fermata)   \n",
       "161286            BigBang    175         1.0         착한 사람 (a Good Man)   \n",
       "161287            BigBang    175         1.0                천국 (heaven)   \n",
       "161288            BigBang    175         1.0          하루 하루 (haru Haru)   \n",
       "\n",
       "                                                    Lyric      Genre  \\\n",
       "0       Tá em dúvida. Não sabe se é normal gostar de d...  Sertanejo   \n",
       "1       Eu sou o começo. Sou o Fim. Sou o A e o Z. Eu ...       Rock   \n",
       "2       [Verse One]. I'm the most critically acclaimed...    Hip Hop   \n",
       "3       Se um dia olhar pro lado. Eu não vou mais esta...       Rock   \n",
       "4       Il cuore è andato in guerra ma la vita. non l'...        Pop   \n",
       "...                                                   ...        ...   \n",
       "161284  Tell me why you gotta look at me that way. You...        Pop   \n",
       "161285  Maeumi swineun dosi. eopseul georan geol ara. ...        Pop   \n",
       "161286  Chakhansaramieosseum angeuraetjyo. Geureona na...    Hip Hop   \n",
       "161287  saranghae nan neol gieokhae. HEAVEN. SING IT T...    Hip Hop   \n",
       "161288  YEAH. FINALLY I REALIZED. THAT I’M NOTHING WIT...    Hip Hop   \n",
       "\n",
       "             Idiom  \n",
       "0       PORTUGUESE  \n",
       "1       PORTUGUESE  \n",
       "2          ENGLISH  \n",
       "3       PORTUGUESE  \n",
       "4          ITALIAN  \n",
       "...            ...  \n",
       "161284     ENGLISH  \n",
       "161285        None  \n",
       "161286        None  \n",
       "161287        None  \n",
       "161288        None  \n",
       "\n",
       "[161289 rows x 7 columns]"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Combine both datasets\n",
    "finaldata = sqldf('SELECT a.Artist, a.Songs, a.Popularity, b.Sname, b.Lyric, a.Genre, b.Idiom FROM fileartist a JOIN filelyrics b on a.Link = b.ALink GROUP BY b.SLink ORDER BY SName')\n",
    "\n",
    "finaldata\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "c2e1766e",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Get rid of None field\n",
    "testdata = finaldata.dropna()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "76e134ed",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "158695"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(testdata)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "df6bffb9",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "21f97f41",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Genre</th>\n",
       "      <th>Total_Count</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Funk Carioca</td>\n",
       "      <td>4340</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Hip Hop</td>\n",
       "      <td>18194</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Pop</td>\n",
       "      <td>37520</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Rock</td>\n",
       "      <td>58040</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Samba</td>\n",
       "      <td>12282</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>Sertanejo</td>\n",
       "      <td>28319</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          Genre  Total_Count\n",
       "0  Funk Carioca         4340\n",
       "1       Hip Hop        18194\n",
       "2           Pop        37520\n",
       "3          Rock        58040\n",
       "4         Samba        12282\n",
       "5     Sertanejo        28319"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#How many songs for each genre of music UNBALANCED\n",
    "songct = sqldf('SELECT Genre, COUNT(*) as Total_Count FROM testdata GROUP BY Genre')\n",
    "\n",
    "songct"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "ed00217d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<BarContainer object of 6 artists>"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#create table for How many songs for each genre of music UNBALANCED\n",
    "names = list(songct['Genre'])\n",
    "values = list(songct['Total_Count'])\n",
    "\n",
    "plt.title(\"Num of Songs Per Genre\")\n",
    "plt.bar(names, values)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "2cffa8a9",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Get sum of popularity by genre of music\n",
    "query10 = sqldf('select Songs, Genre, Popularity from testdata order by Popularity desc')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "a95c8b3e",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Get sum of popularity by genre of music\n",
    "querypop = sqldf('select sum(Popularity) as sum_pop, Genre from query10 group by genre')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "29f56b59",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sum_pop</th>\n",
       "      <th>Genre</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>5727.7</td>\n",
       "      <td>Funk Carioca</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>99648.4</td>\n",
       "      <td>Hip Hop</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>292820.5</td>\n",
       "      <td>Pop</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>252171.3</td>\n",
       "      <td>Rock</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>48876.2</td>\n",
       "      <td>Samba</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>118477.0</td>\n",
       "      <td>Sertanejo</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    sum_pop         Genre\n",
       "0    5727.7  Funk Carioca\n",
       "1   99648.4       Hip Hop\n",
       "2  292820.5           Pop\n",
       "3  252171.3          Rock\n",
       "4   48876.2         Samba\n",
       "5  118477.0     Sertanejo"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "display(querypop)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "3647549a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<BarContainer object of 6 artists>"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "names = list(querypop['Genre'])\n",
    "values = list(querypop['sum_pop'])\n",
    "\n",
    "plt.title(\"Popularity Score Per Genre of Music\")\n",
    "plt.bar(names, values)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7b83e04c",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "46d010e5",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "23e1e56c",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Time to preprocess data. Function created to do so"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "73573f19",
   "metadata": {},
   "outputs": [],
   "source": [
    "def vectorize(keywords):\n",
    "\n",
    "    #------------------------------------------------------------------------------------------------------------\n",
    "    ## Lowercase Words\n",
    "    #------------------------------------------------------------------------------------------------------------\n",
    "    \n",
    "    keywords = [w.lower( ) for w in keywords]\n",
    "    #print(keywords[:10])\n",
    "    \n",
    "    #get count of total words\n",
    "    print(\"total words before vectorization: \", len(keywords))\n",
    "    \n",
    "    #---------------------------\n",
    "    ## Add Stemming\n",
    "    #------------------------------\n",
    "    \n",
    "    #Time to stem words together    \n",
    "    ps = PorterStemmer()   ## method from nltk\n",
    "    \n",
    "    stemmed_words=[]  ## make new empty list\n",
    "    for w in keywords:\n",
    "        stemmed_words.append(ps.stem(w))\n",
    "        \n",
    "    #get count of total words after stemming\n",
    "    print(\"total words after stemming (Should be the same): \", len(stemmed_words))\n",
    "        \n",
    "        \n",
    "    #------------------------------------------------------------------------------------------------------------\n",
    "    ## Removing Stopwords\n",
    "    #------------------------------------------------------------------------------------------------------------\n",
    "    \n",
    "    #Get NLTK stop words\n",
    "    stop_words=set(stopwords.words(\"english\"))\n",
    "    \n",
    "    stop_words_span = set(stopwords.words(\"spanish\"))\n",
    "    \n",
    "    stop_words_port = set(stopwords.words(\"portuguese\"))\n",
    "    \n",
    "    stop_words_germ = set(stopwords.words(\"german\"))\n",
    "    \n",
    "    morestopwords = set(['\\'s', '\\'nt', 'nt\\'', 'nt', 'i', 'n\\'t', '\\'m', '\\'ll', '\\'re', 'ca', '\\'ve', 'oh', 'yeah', 'a', 'e', 'i', 'o', 'u', 'im', 'thi', 'hi', '\\'na', 'na\\'', 'na', 'n\\'a'])\n",
    "    \n",
    "    stop_words = stop_words.union(morestopwords)\n",
    "    \n",
    "    stop_words = stop_words.union(stop_words_span)\n",
    "    \n",
    "    stop_words = stop_words.union(stop_words_port)\n",
    "    \n",
    "    stop_words = stop_words.union(stop_words_germ)\n",
    "     \n",
    "    #print(stop_words)\n",
    "    \n",
    "    filtered_text=[]   ## Create a new empty list\n",
    "    \n",
    "    for w in stemmed_words:\n",
    "        #print(w)\n",
    "        if w not in stop_words:\n",
    "            filtered_text.append(w)\n",
    "        \n",
    "    #get count of total words after removing stopwords\n",
    "    print(\"total words after removing stopwords: \", len(filtered_text))\n",
    "    \n",
    "    \n",
    "    #------------------------------------------------------------------------------------------------------------\n",
    "    ## Remove Punctuation\n",
    "    #------------------------------------------------------------------------------------------------------------\n",
    "    \n",
    "    #Check punctuation\n",
    "    #print(string.punctuation)\n",
    "    \n",
    "    #Any reference to punctuation turn into blank\n",
    "    table = str.maketrans('', '', string.punctuation)\n",
    "    stripped = [w.translate(table) for w in filtered_text]\n",
    "    #print(stripped[:100])\n",
    "    \n",
    "    #get count of total words after removing punctuation\n",
    "    #print(\"total words after removing punctuation: \", len(stripped))\n",
    "    \n",
    "    #------------------------------------------------------------------------------------------------------------\n",
    "    ## Remove Empty Strings\n",
    "    #------------------------------------------------------------------------------------------------------------\n",
    "    \n",
    "    #remove empty strings \n",
    "    while(\"\" in stripped) :\n",
    "        stripped.remove(\"\")\n",
    "        \n",
    "    #get count of total words after removing empty strings\n",
    "    print(\"total words after removing punctuation/empty strings: \", len(stripped))\n",
    "        \n",
    "    return(stripped)\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "c21a6749",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "total words before vectorization:  158695\n",
      "total words after stemming (Should be the same):  158695\n",
      "total words after removing stopwords:  158695\n",
      "total words after removing punctuation/empty strings:  158695\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "<ipython-input-26-b239be5f6fc9>:1: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  testdata['Lyric'] = vectorize(testdata['Lyric'])\n"
     ]
    }
   ],
   "source": [
    "testdata['Lyric'] = vectorize(testdata['Lyric'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "3d26ff36",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Artist</th>\n",
       "      <th>Songs</th>\n",
       "      <th>Popularity</th>\n",
       "      <th>SName</th>\n",
       "      <th>Lyric</th>\n",
       "      <th>Genre</th>\n",
       "      <th>Idiom</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Luan Santana</td>\n",
       "      <td>187</td>\n",
       "      <td>17.2</td>\n",
       "      <td>\"A\"</td>\n",
       "      <td>tá em dúvida não sabe se é normal gostar de do...</td>\n",
       "      <td>Sertanejo</td>\n",
       "      <td>PORTUGUESE</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Mutantes</td>\n",
       "      <td>123</td>\n",
       "      <td>1.0</td>\n",
       "      <td>\"A\" e o \"Z\"</td>\n",
       "      <td>eu sou o começo sou o fim sou o a e o z eu sou...</td>\n",
       "      <td>Rock</td>\n",
       "      <td>PORTUGUESE</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Foxy Brown</td>\n",
       "      <td>74</td>\n",
       "      <td>0.0</td>\n",
       "      <td>\"Oh Yeah\" By Foxy Brown</td>\n",
       "      <td>verse one im the most critically acclaimed rap...</td>\n",
       "      <td>Hip Hop</td>\n",
       "      <td>ENGLISH</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Barbie Kue</td>\n",
       "      <td>8</td>\n",
       "      <td>0.0</td>\n",
       "      <td>\"Sei Lá...\"</td>\n",
       "      <td>se um dia olhar pro lado eu não vou mais estar...</td>\n",
       "      <td>Rock</td>\n",
       "      <td>PORTUGUESE</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Tiziano Ferro</td>\n",
       "      <td>160</td>\n",
       "      <td>2.1</td>\n",
       "      <td>\"Solo\" e' Solo Una Parola</td>\n",
       "      <td>il cuore è andato in guerra ma la vita non lho...</td>\n",
       "      <td>Pop</td>\n",
       "      <td>ITALIAN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>161280</th>\n",
       "      <td>MV Bill</td>\n",
       "      <td>92</td>\n",
       "      <td>1.7</td>\n",
       "      <td>é Nós E A Gente</td>\n",
       "      <td>a paz não precisa ser um sonho basta o respeit...</td>\n",
       "      <td>Hip Hop</td>\n",
       "      <td>PORTUGUESE</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>161281</th>\n",
       "      <td>Furacão 2000</td>\n",
       "      <td>57</td>\n",
       "      <td>2.8</td>\n",
       "      <td>é O Kit</td>\n",
       "      <td>é o kit é o kit furacão se liga mané é o kit t...</td>\n",
       "      <td>Funk Carioca</td>\n",
       "      <td>PORTUGUESE</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>161282</th>\n",
       "      <td>Fundo de Quintal</td>\n",
       "      <td>315</td>\n",
       "      <td>4.6</td>\n",
       "      <td>ô Irene</td>\n",
       "      <td>ô irene ô irene ô irene ô irene vai buscar o q...</td>\n",
       "      <td>Samba</td>\n",
       "      <td>PORTUGUESE</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>161283</th>\n",
       "      <td>Vremya I Steklo</td>\n",
       "      <td>5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>На Стилe</td>\n",
       "      <td>а мы на стиле даже и не думай чтото отменять у...</td>\n",
       "      <td>Pop</td>\n",
       "      <td>RUSSIAN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>161284</th>\n",
       "      <td>Ariana Grande</td>\n",
       "      <td>155</td>\n",
       "      <td>246.8</td>\n",
       "      <td>​Goodnight n Go</td>\n",
       "      <td>tell me why you gotta look at me that way you ...</td>\n",
       "      <td>Pop</td>\n",
       "      <td>ENGLISH</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>158695 rows × 7 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                  Artist  Songs  Popularity                      SName  \\\n",
       "0           Luan Santana    187        17.2                        \"A\"   \n",
       "1               Mutantes    123         1.0                \"A\" e o \"Z\"   \n",
       "2             Foxy Brown     74         0.0    \"Oh Yeah\" By Foxy Brown   \n",
       "3             Barbie Kue      8         0.0                \"Sei Lá...\"   \n",
       "4          Tiziano Ferro    160         2.1  \"Solo\" e' Solo Una Parola   \n",
       "...                  ...    ...         ...                        ...   \n",
       "161280           MV Bill     92         1.7            é Nós E A Gente   \n",
       "161281      Furacão 2000     57         2.8                    é O Kit   \n",
       "161282  Fundo de Quintal    315         4.6                    ô Irene   \n",
       "161283   Vremya I Steklo      5         0.0                   На Стилe   \n",
       "161284     Ariana Grande    155       246.8            ​Goodnight n Go   \n",
       "\n",
       "                                                    Lyric         Genre  \\\n",
       "0       tá em dúvida não sabe se é normal gostar de do...     Sertanejo   \n",
       "1       eu sou o começo sou o fim sou o a e o z eu sou...          Rock   \n",
       "2       verse one im the most critically acclaimed rap...       Hip Hop   \n",
       "3       se um dia olhar pro lado eu não vou mais estar...          Rock   \n",
       "4       il cuore è andato in guerra ma la vita non lho...           Pop   \n",
       "...                                                   ...           ...   \n",
       "161280  a paz não precisa ser um sonho basta o respeit...       Hip Hop   \n",
       "161281  é o kit é o kit furacão se liga mané é o kit t...  Funk Carioca   \n",
       "161282  ô irene ô irene ô irene ô irene vai buscar o q...         Samba   \n",
       "161283  а мы на стиле даже и не думай чтото отменять у...           Pop   \n",
       "161284  tell me why you gotta look at me that way you ...           Pop   \n",
       "\n",
       "             Idiom  \n",
       "0       PORTUGUESE  \n",
       "1       PORTUGUESE  \n",
       "2          ENGLISH  \n",
       "3       PORTUGUESE  \n",
       "4          ITALIAN  \n",
       "...            ...  \n",
       "161280  PORTUGUESE  \n",
       "161281  PORTUGUESE  \n",
       "161282  PORTUGUESE  \n",
       "161283     RUSSIAN  \n",
       "161284     ENGLISH  \n",
       "\n",
       "[158695 rows x 7 columns]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "display(testdata)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "001bcafe",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "555a0107",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = testdata['Lyric'].values\n",
    "Y = testdata['Genre'].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "d1131e62",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(95217,) (95217,) (63478,) (63478,)\n",
      "fiquei no sofá esperando você ainda bem que eu sentei se não além do coração a perna ia doer doeu não fiquei com ninguém primeiro porque quem se aproximou de mim tava com dó do que você aprontou comigo graças a deus arranjei um monte de amigo um monte de conselho um pouco de vergonha na cara 3 doses de whisky do tempo que o cigarro apaga esse amor acaba e some igual fumaça se até o tempo passa imagina o seu amor devolve o meu tempo que eu fiquei ligando mandando mensagem implorando por favor vou te cobrar com juros o que você me tirou se até o tempo passa imagina o nosso amor e esse arranhadinho no meu coração soprou passou passou e o tempo levou se é pra viver sofrendo eu vou sofrer com um novo amor\n",
      "Sertanejo\n",
      "eu não sei o que acontece comigo você era só meu amigo não sei se te passa o mesmo e também se amarrou tanto em mim eu sei o que se passa comigo meu corpo não sabe mentir te olho te chamo te sinto na pele não deixo um segundo de pensar em ti refrão estou louco louca fiquei louco louca num segundo fugiste de mim e eu penso em você te imagino correndo cabelos ao vento me beija dizendo que louco estou louco por ti não sei o que será do amanhã hoje é um dia especial ainda em meu corpo seu beijo me queima mais que o sol eu sei o que será dessa espera nós dois na mesma emoção sentir sua pele incendiar o meu corpo seremos um só coração repete refrão 2x\n",
      "Pop\n"
     ]
    }
   ],
   "source": [
    "#use bayes to train adn test using CV\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=0)\n",
    "\n",
    "print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)\n",
    "print(X_train[0])\n",
    "print(Y_train[0])\n",
    "print(X_test[0])\n",
    "print(Y_test[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "b1e668fe",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(95217,) (95217,) (63478,) (63478,)\n"
     ]
    }
   ],
   "source": [
    "print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "28993574",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[['Funk Carioca' 'Hip Hop' 'Pop' 'Rock' 'Samba' 'Sertanejo']\n",
      " [2589 11011 22447 34998 7217 16955]]\n"
     ]
    }
   ],
   "source": [
    "# Check how many training examples in each category\n",
    "unique, counts = np.unique(Y_train, return_counts=True)\n",
    "print(np.asarray((unique, counts)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "f721b3f9",
   "metadata": {},
   "outputs": [],
   "source": [
    "#  unigram boolean vectorizer, set minimum document frequency to 5\n",
    "unigram_bool_vectorizer = CountVectorizer(encoding='latin-1', binary=True, min_df=5, stop_words='english')\n",
    "\n",
    "X_train_vec = unigram_bool_vectorizer.fit_transform(X_train)\n",
    "\n",
    "# use the vocabulary constructed from the training data to vectorize the test data. \n",
    "X_test_vec = unigram_bool_vectorizer.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "4f65511b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "MultinomialNB()"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# import the MNB module\n",
    "from sklearn.naive_bayes import MultinomialNB\n",
    "\n",
    "# initialize the MNB model\n",
    "nb_clf= MultinomialNB()\n",
    "\n",
    "# use the training data to train the MNB model\n",
    "nb_clf.fit(X_train_vec,Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "9d75d409",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.629572450297741"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# test the classifier on the test data set, print accuracy score\n",
    "nb_clf.score(X_test_vec,Y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "2fca0c33",
   "metadata": {},
   "outputs": [],
   "source": [
    "#predict NB models\n",
    "Y_pred = nb_clf.fit(X_train_vec, Y_train).predict(X_test_vec)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "5431d703",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.7104980621562674\n"
     ]
    }
   ],
   "source": [
    "# cross validation\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.model_selection import cross_val_score\n",
    "nb_clf_pipe = Pipeline([('vect', CountVectorizer(encoding='latin-1', binary=False)),('nb', MultinomialNB())])\n",
    "scores = cross_val_score(nb_clf_pipe, Y_test, Y_pred, cv=3)\n",
    "avg=sum(scores)/len(scores)\n",
    "print(avg)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "9b89ea18",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "       Samba       0.45      0.73      0.56      1751\n",
      "Funk Carioca       0.80      0.55      0.65      7183\n",
      "     Hip Hop       0.69      0.20      0.31     15073\n",
      "   Sertanejo       0.67      0.80      0.73     23042\n",
      "         Pop       0.53      0.50      0.52      5065\n",
      "        Rock       0.57      0.94      0.71     11364\n",
      "\n",
      "    accuracy                           0.63     63478\n",
      "   macro avg       0.62      0.62      0.58     63478\n",
      "weighted avg       0.65      0.63      0.60     63478\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# print accuracies and classification report\n",
    "from sklearn.metrics import classification_report\n",
    "target_names = [\"Samba\", \"Funk Carioca\", \"Hip Hop\", \"Sertanejo\", \"Pop\", \"Rock\"]\n",
    "print(classification_report(Y_test, Y_pred, target_names=target_names))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "a1673f15",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                P:Samba  P:Funk Carioca  P:Hip Hop  P:Sertanejo  P:Pop  P:Rock\n",
      "T:Samba            2546              95          0         2344     67      13\n",
      "T:Funk Carioca       77            1280          3          385      3       3\n",
      "T:Hip Hop           268             856       3976          419    738     926\n",
      "T:Sertanejo         315             275          0        10683     69      22\n",
      "T:Pop               700             194        697         2293   3026    8163\n",
      "T:Rock              873             152        306         2753    505   18453\n"
     ]
    }
   ],
   "source": [
    "#confusion martix\n",
    "df = pd.DataFrame(\n",
    "    confusion_matrix(Y_test, Y_pred, labels=[\"Samba\", \"Funk Carioca\", \"Hip Hop\", \"Sertanejo\", \"Pop\", \"Rock\"]),\n",
    "    index = ['T:Samba', 'T:Funk Carioca', 'T:Hip Hop', 'T:Sertanejo', 'T:Pop', 'T:Rock'], \n",
    "    columns = ['P:Samba', 'P:Funk Carioca', 'P:Hip Hop', 'P:Sertanejo', 'P:Pop', 'P:Rock']\n",
    ")\n",
    "print(df)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7b5f1340",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1706924e",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "d8c4cd75",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/alec_arroyo/opt/anaconda3/lib/python3.8/site-packages/sklearn/svm/_base.py:1206: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "LinearSVC(C=1)"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# import the LinearSVC module\n",
    "from sklearn.svm import LinearSVC\n",
    "\n",
    "# initialize the LinearSVC model\n",
    "svm_clf = LinearSVC(C=1)\n",
    "\n",
    "# use the training data to train the model\n",
    "svm_clf.fit(X_train_vec,Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "cdae3260",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.6816377327578058"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# test the classifier on the test data set, print accuracy score\n",
    "svm_clf.score(X_test_vec,Y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "ef72b28c",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/alec_arroyo/opt/anaconda3/lib/python3.8/site-packages/sklearn/svm/_base.py:1206: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.\n",
      "  warnings.warn(\n"
     ]
    }
   ],
   "source": [
    "#predict the SVM model\n",
    "Y_pred = svm_clf.fit(X_train_vec, Y_train).predict(X_test_vec)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "1a0224ee",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "       Samba       0.72      0.66      0.69      1751\n",
      "Funk Carioca       0.79      0.72      0.75      7183\n",
      "     Hip Hop       0.53      0.53      0.53     15073\n",
      "   Sertanejo       0.72      0.74      0.73     23042\n",
      "         Pop       0.61      0.58      0.60      5065\n",
      "        Rock       0.76      0.79      0.78     11364\n",
      "\n",
      "    accuracy                           0.68     63478\n",
      "   macro avg       0.69      0.67      0.68     63478\n",
      "weighted avg       0.68      0.68      0.68     63478\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# print confusion matrix and classification report\n",
    "from sklearn.metrics import classification_report\n",
    "target_names = [\"Samba\", \"Funk Carioca\", \"Hip Hop\", \"Sertanejo\", \"Pop\", \"Rock\"]\n",
    "print(classification_report(Y_test, Y_pred, target_names=target_names))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "331e7502",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.6815117255307007\n"
     ]
    }
   ],
   "source": [
    "# cross validation\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.model_selection import cross_val_score\n",
    "svm_clf_pipe = Pipeline([('vect', CountVectorizer(encoding='latin-1', binary=False)),('svm', LinearSVC(C=1))])\n",
    "scores = cross_val_score(svm_clf_pipe, Y_test, Y_pred, cv=3)\n",
    "avg=sum(scores)/len(scores)\n",
    "print(avg)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "036b8432",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                P:Samba  P:Funk Carioca  P:Hip Hop  P:Sertanejo  P:Pop  P:Rock\n",
      "T:Samba            2956              73         66         1064    504     402\n",
      "T:Funk Carioca      123            1153         59          183    153      80\n",
      "T:Hip Hop            71              69       5140          100   1103     700\n",
      "T:Sertanejo         862             151         40         8994    792     525\n",
      "T:Pop               437              94        792          841   7957    4952\n",
      "T:Rock              394              54        375          616   4542   17061\n"
     ]
    }
   ],
   "source": [
    "#confusion martix\n",
    "df = pd.DataFrame(\n",
    "    confusion_matrix(Y_test, Y_pred, labels=[\"Samba\", \"Funk Carioca\", \"Hip Hop\", \"Sertanejo\", \"Pop\", \"Rock\"]),\n",
    "    index = ['T:Samba', 'T:Funk Carioca', 'T:Hip Hop', 'T:Sertanejo', 'T:Pop', 'T:Rock'], \n",
    "    columns = ['P:Samba', 'P:Funk Carioca', 'P:Hip Hop', 'P:Sertanejo', 'P:Pop', 'P:Rock']\n",
    ")\n",
    "print(df)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c28c1f14",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b211b74c",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5e05ce44",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "d7f00faa",
   "metadata": {},
   "outputs": [],
   "source": [
    "#perform Oversampling to deal w/ unbalanced data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "600201cc",
   "metadata": {},
   "outputs": [],
   "source": [
    "newbalanceddata = testdata"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "a9e6d18c",
   "metadata": {},
   "outputs": [],
   "source": [
    "bal_X = pd.DataFrame(newbalanceddata.Lyric)\n",
    "bal_Y = pd.DataFrame(newbalanceddata.Genre)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "79bf808d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>count(*)</th>\n",
       "      <th>Genre</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>4340</td>\n",
       "      <td>Funk Carioca</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>18194</td>\n",
       "      <td>Hip Hop</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>37520</td>\n",
       "      <td>Pop</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>58040</td>\n",
       "      <td>Rock</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>12282</td>\n",
       "      <td>Samba</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>28319</td>\n",
       "      <td>Sertanejo</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   count(*)         Genre\n",
       "0      4340  Funk Carioca\n",
       "1     18194       Hip Hop\n",
       "2     37520           Pop\n",
       "3     58040          Rock\n",
       "4     12282         Samba\n",
       "5     28319     Sertanejo"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sqldf('select count(*), Genre from bal_Y group by Genre')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "c88d874d",
   "metadata": {},
   "outputs": [],
   "source": [
    "#most important part\n",
    "ran = RandomOverSampler(sampling_strategy = 'not majority')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "159c21d6",
   "metadata": {},
   "outputs": [],
   "source": [
    "ran_X, ran_Y = ran.fit_resample(bal_X, bal_Y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "56862939",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>count(*)</th>\n",
       "      <th>Genre</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>58040</td>\n",
       "      <td>Funk Carioca</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>58040</td>\n",
       "      <td>Hip Hop</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>58040</td>\n",
       "      <td>Pop</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>58040</td>\n",
       "      <td>Rock</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>58040</td>\n",
       "      <td>Samba</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>58040</td>\n",
       "      <td>Sertanejo</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   count(*)         Genre\n",
       "0     58040  Funk Carioca\n",
       "1     58040       Hip Hop\n",
       "2     58040           Pop\n",
       "3     58040          Rock\n",
       "4     58040         Samba\n",
       "5     58040     Sertanejo"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sqldf('select count(*), Genre from ran_Y group by Genre')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "dab09601",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'ran_X' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-2-fec1a09d1d50>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;31m#use bayes to train and test BALANCED DATASET\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 3\u001b[0;31m \u001b[0mran_X\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mran_X\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m'Lyric'\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mvalues\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      4\u001b[0m \u001b[0mran_Y\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mran_Y\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m'Genre'\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mvalues\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mNameError\u001b[0m: name 'ran_X' is not defined"
     ]
    }
   ],
   "source": [
    "#use bayes to train and test BALANCED DATASET\n",
    "\n",
    "ran_X = ran_X['Lyric'].values\n",
    "ran_Y = ran_Y['Genre'].values\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, Y_train, Y_test = train_test_split(ran_X, ran_Y, test_size=0.4, random_state=0)\n",
    "\n",
    "print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)\n",
    "\n",
    "# Check how many training examples in each category\n",
    "unique, counts = np.unique(Y_train, return_counts=True)\n",
    "print(np.asarray((unique, counts)))\n",
    "\n",
    "\n",
    "#  unigram boolean vectorizer, set minimum document frequency to 5\n",
    "unigram_bool_vectorizer = CountVectorizer(encoding='latin-1', binary=True, min_df=5, stop_words='english')\n",
    "\n",
    "X_train_vec = unigram_bool_vectorizer.fit_transform(X_train)\n",
    "\n",
    "# use the vocabulary constructed from the training data to vectorize the test data. \n",
    "X_test_vec = unigram_bool_vectorizer.transform(X_test)\n",
    "\n",
    "\n",
    "# import the MNB module\n",
    "from sklearn.naive_bayes import MultinomialNB\n",
    "\n",
    "# initialize the MNB model\n",
    "nb_clf= MultinomialNB()\n",
    "\n",
    "# use the training data to train the MNB model\n",
    "nb_clf.fit(X_train_vec,Y_train)\n",
    "\n",
    "\n",
    "# test the classifier on the test data set, print accuracy score\n",
    "print(nb_clf.score(X_test_vec,Y_test))\n",
    "\n",
    "#predict the NB model\n",
    "Y_pred = nb_clf.fit(X_train_vec, Y_train).predict(X_test_vec)\n",
    "\n",
    "\n",
    "# print confusion matrix and classification report\n",
    "\n",
    "from sklearn.metrics import classification_report\n",
    "target_names = [\"Samba\", \"Funk Carioca\", \"Hip Hop\", \"Sertanejo\", \"Pop\", \"Rock\"]\n",
    "print(classification_report(Y_test, Y_pred, target_names=target_names))\n",
    "\n",
    "\n",
    "#confusion martix\n",
    "df = pd.DataFrame(\n",
    "    confusion_matrix(Y_test, Y_pred, labels=[\"Samba\", \"Funk Carioca\", \"Hip Hop\", \"Sertanejo\", \"Pop\", \"Rock\"]),\n",
    "    index = ['T:Samba', 'T:Funk Carioca', 'T:Hip Hop', 'T:Sertanejo', 'T:Pop', 'T:Rock'], \n",
    "    columns = ['P:Samba', 'P:Funk Carioca', 'P:Hip Hop', 'P:Sertanejo', 'P:Pop', 'P:Rock']\n",
    ")\n",
    "print(df)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "538373e5",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "947141d1",
   "metadata": {},
   "outputs": [],
   "source": [
    "#repeat same process for SVM\n",
    "\n",
    "newbalanceddata = testdata\n",
    "\n",
    "bal_X = pd.DataFrame(newbalanceddata.Lyric)\n",
    "bal_Y = pd.DataFrame(newbalanceddata.Genre)\n",
    "\n",
    "#most important part\n",
    "ran = RandomOverSampler(sampling_strategy = 'not majority')\n",
    "\n",
    "ran_X, ran_Y = ran.fit_resample(bal_X, bal_Y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "fa3f33ae",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(208944,) (208944,) (139296,) (139296,)\n",
      "[['Funk Carioca' 'Hip Hop' 'Pop' 'Rock' 'Samba' 'Sertanejo']\n",
      " [34889 34757 34799 34710 35033 34756]]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/alec_arroyo/opt/anaconda3/lib/python3.8/site-packages/sklearn/svm/_base.py:1206: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.8725950493912245\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/alec_arroyo/opt/anaconda3/lib/python3.8/site-packages/sklearn/svm/_base.py:1206: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "       Samba       0.98      1.00      0.99     23151\n",
      "Funk Carioca       0.93      0.95      0.94     23283\n",
      "     Hip Hop       0.72      0.71      0.71     23241\n",
      "   Sertanejo       0.76      0.69      0.73     23330\n",
      "         Pop       0.91      0.97      0.94     23007\n",
      "        Rock       0.91      0.92      0.91     23284\n",
      "\n",
      "    accuracy                           0.87    139296\n",
      "   macro avg       0.87      0.87      0.87    139296\n",
      "weighted avg       0.87      0.87      0.87    139296\n",
      "\n",
      "                P:Samba  P:Funk Carioca  P:Hip Hop  P:Sertanejo  P:Pop  P:Rock\n",
      "T:Samba           22413              28         13          363    112      78\n",
      "T:Funk Carioca        0           23151          0            0      0       0\n",
      "T:Hip Hop            56              49      22030           59    655     434\n",
      "T:Sertanejo         908             144         40        21329    546     317\n",
      "T:Pop               600             123        923          883  16465    4247\n",
      "T:Rock              519              87        614          711   5242   16157\n"
     ]
    }
   ],
   "source": [
    "#use SVM to train and test BALANCED DATASET\n",
    "\n",
    "ran_X = ran_X['Lyric'].values\n",
    "ran_Y = ran_Y['Genre'].values\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, Y_train, Y_test = train_test_split(ran_X, ran_Y, test_size=0.4, random_state=0)\n",
    "\n",
    "print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)\n",
    "\n",
    "\n",
    "\n",
    "# Check how many training examples in each category\n",
    "unique, counts = np.unique(Y_train, return_counts=True)\n",
    "print(np.asarray((unique, counts)))\n",
    "\n",
    "\n",
    "#  unigram boolean vectorizer, set minimum document frequency to 5\n",
    "unigram_bool_vectorizer = CountVectorizer(encoding='latin-1', binary=True, min_df=5, stop_words='english')\n",
    "\n",
    "X_train_vec = unigram_bool_vectorizer.fit_transform(X_train)\n",
    "\n",
    "# use the vocabulary constructed from the training data to vectorize the test data. \n",
    "X_test_vec = unigram_bool_vectorizer.transform(X_test)\n",
    "\n",
    "\n",
    "\n",
    "# import the LinearSVC module\n",
    "from sklearn.svm import LinearSVC\n",
    "\n",
    "# initialize the LinearSVC model\n",
    "svm_clf = LinearSVC(C=1)\n",
    "\n",
    "# use the training data to train the model\n",
    "svm_clf.fit(X_train_vec,Y_train)\n",
    "\n",
    "\n",
    "# test the classifier on the test data set, print accuracy score\n",
    "print(svm_clf.score(X_test_vec,Y_test))\n",
    "\n",
    "\n",
    "#predict the SVM model\n",
    "Y_pred = svm_clf.fit(X_train_vec, Y_train).predict(X_test_vec)\n",
    "\n",
    "\n",
    "# print confusion matrix and classification report\n",
    "from sklearn.metrics import classification_report\n",
    "target_names = [\"Samba\", \"Funk Carioca\", \"Hip Hop\", \"Sertanejo\", \"Pop\", \"Rock\"]\n",
    "print(classification_report(Y_test, Y_pred, target_names=target_names))\n",
    "\n",
    "\n",
    "#confusion martix\n",
    "df = pd.DataFrame(\n",
    "    confusion_matrix(Y_test, Y_pred, labels=[\"Samba\", \"Funk Carioca\", \"Hip Hop\", \"Sertanejo\", \"Pop\", \"Rock\"]),\n",
    "    index = ['T:Samba', 'T:Funk Carioca', 'T:Hip Hop', 'T:Sertanejo', 'T:Pop', 'T:Rock'], \n",
    "    columns = ['P:Samba', 'P:Funk Carioca', 'P:Hip Hop', 'P:Sertanejo', 'P:Pop', 'P:Rock']\n",
    ")\n",
    "print(df)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "d3473afb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "numpy.ndarray"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "5b828a2c",
   "metadata": {},
   "outputs": [],
   "source": [
    "newArray = ['Buddy, youre a boy, make a big noise Playing in the street, gonna be a big man someday You got mud on your face, you big disgrace Kicking your can all over the place, singin We will, we will rock you We will, we will rock you Buddy, youre a young man, hard man Shouting in the street, gonna take on the world someday You got blood on your face, you big disgrace Waving your banner all over the place We will, we will rock you, sing it! We will, we will rock you, yeah Buddy, youre an old man, poor man Pleading with your eyes, gonna get you some peace someday You got mud on your face, big disgrace Somebody better put you back into your place, do it! We will, we will rock you, yeah, yeah, come on We will, we will rock you, alright, louder! We will, we will rock you, one more time We will, we will rock you Yeah']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "872219fa",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "25ae2583",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7894f2de",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Intake new song and predict what genre of music it is"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "d2cd66e1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'Buddy, youre a boy, make a big noise Playing in the street, gonna be a big man someday You got mud on your face, you big disgrace Kicking your can all over the place, singin We will, we will rock you We will, we will rock you Buddy, youre a young man, hard man Shouting in the street, gonna take on the world someday You got blood on your face, you big disgrace Waving your banner all over the place We will, we will rock you, sing it! We will, we will rock you, yeah Buddy, youre an old man, poor man Pleading with your eyes, gonna get you some peace someday You got mud on your face, big disgrace Somebody better put you back into your place, do it! We will, we will rock you, yeah, yeah, come on We will, we will rock you, alright, louder! We will, we will rock you, one more time We will, we will rock you Yeah'"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "newArray"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "0b419d7f",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Transform to fit model\n",
    "X_test_vecARRAY = unigram_bool_vectorizer.transform(newArray)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "f0749f80",
   "metadata": {},
   "outputs": [],
   "source": [
    "#predict Lyrics to genre\n",
    "Y_pred = nb_clf.fit(X_train_vec, Y_train).predict(X_test_vecARRAY)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "8baeed78",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['Rock'], dtype='<U12')"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Genre prediction was correct. this is a rock song\n",
    "Y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "86474dd5",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "43504adb",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "54bbe860",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Enter your song lyrics for one song here: Ain't this what they've been waitin' for? You ready? Uh, uh I used to pray for times like this, to rhyme like this So I had to grind like that to shine like this In a matter of time I spent on some locked-up shit In the back of the paddy wagon, cuffs locked on wrists Seen my dreams unfold, nightmares come true It was time to marry the game and I said, \"Yeah, I do.\" If you want it, you gotta see it with a clear-eyed view Got a shorty, she tryna bless me like I said achoo Like a nigga sneezed, nigga please, 'fore them triggers squeeze I'm gettin' cream, never let them hoes get in between Of what we started, lil' nigga but I'm lion-hearted They love me when I was stuck and they hated when I departed I go and get it regardless, draw it like I'm a artist No crawlin', went straight to walkin', with foreigns in my garages All foreign bitches ménagin', fuckin', suckin' and swallowin' Anything for a dollar, they tell me get 'em, I got 'em I did it without a album I did shit with Mariah Lil' nigga, I'm on fire Icy as a hockey rink, Philly nigga, I'm flyer When I bought the Rolls-Royce they thought it was leased Then I bought that new Ferrari, hater, rest in peace Hater, rest in peace, rest in peace to the parking lot Phantom so big, can't even fit in the parking spot You ain't talkin' 'bout my niggas, then what you talkin' 'bout? Gangstas move in silence, nigga, and I don't talk a lot I don't say a word, I don't say a word Was on my grind and now I got what I deserve, fuck nigga Hold up, wait a minute, y'all thought I was finished? When I bought that Aston Martin, y'all thought it was rented? Flexin' on these niggas, I'm like Popeye on his spinach Double M, yeah, that's my team, Rozay the captain, I'm lieutenant I'm the type to count a million cash then grind like I'm broke That Lambo my new bitch, she don't ride like my Ghost I'm ridin' around my city with my hand strapped on my toast 'Cause these niggas want me dead and I gotta make it back home 'Cause my mama need that bill money, my son need some milk These niggas tryna take my life, they fuck around, get killed You fuck around, you fuck around, you fuck around, get smoked 'Cause these Philly niggas I brought with me don't fuck around, no joke, no All I know is murder, when it come to me I got young niggas that's rollin', I got niggas throwin' B's I done did the DOA's, I done did the KOD's Every time I'm in that bitch, I get to throwin' 30 G's But now I'm hangin' out that drop head, I'm ridin' down on Collins They let my nigga Ern back home, that young nigga be wildin' We young niggas, we mobbin', like Batman and we're robbin' This two-door Maybach with my seat all reclinin' I'm like, \"Real nigga, what up? Real nigga, what up?\" If you ain't about that murder game, then pussy nigga, shut up If you diss me in your raps I'll get yo' pussy-ass stuck up When you touchdown in my hood, no that tour life ain't good Catch me down in MIA at that Heat game on wood With that Puma life on my feet like that little engine I could Boy, I slide down on your block, bike on 12 o'clock And they be throwin' deuces on the same nigga they watch And I'm the king of my city 'cause I'm still callin' them shots And these lames talkin' that bullshit the same niggas that flopped I'm the same nigga from Berks Street with them nappy braids, that lock The same nigga that came up and I had to wait for my spot And these niggas hatin' on me, hoes waitin' on me Still on that hood shit, my Rolls-Royce on E They gon' remember me, I say remember me So much money, have yo' friends turn in yo' enemies And when there's beef I turn my enemies to memories With them bricks they go for 40, ain't no 10 a key Hold up, broke nigga turned rich, love the game like Mitch And if I leave, you think them pretty hoes gon' still suck my dick? It was somethin' about that Rollie when it first touched my wrist Had me feelin' like that dope boy when he first touched that brick I'm gone Woo\n"
     ]
    }
   ],
   "source": [
    "#Prompt user to input song lyrics\n",
    "newlyric = input(\"Enter your song lyrics for one song here: \")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "993a8ec0",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Convert lyrics variable into a list\n",
    "newlyric1 = []\n",
    "newlyric1 = [newlyric]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "033c1444",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Transform to fit model\n",
    "X_test_vecARRAY = unigram_bool_vectorizer.transform(newlyric1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "9fd4f880",
   "metadata": {},
   "outputs": [],
   "source": [
    "#predict Lyrics to genre\n",
    "Y_pred = nb_clf.fit(X_train_vec, Y_train).predict(X_test_vecARRAY)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "2d3a5988",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['Hip Hop'], dtype='<U12')"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Genre prediction was correct. this is a Hip Hop song\n",
    "Y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fd91946b",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
