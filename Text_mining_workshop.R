# Test Mining with R 
# March 5, 2014
# StatLab@UVa Library
# Clay Ford



# need to provide workshop attendees zip file with directory structure;
# probably need to provide instructions about where to unzip;
# test that out on your computer!


# Corpus/TDM Example ------------------------------------------------------

# toy example of turning text into numbers
setwd("~/workshops/Text Mining/docs")
# save files name to vector
(files <- dir())

# create a vector to store content of files
allData <- rep(NA,length(files))

# a for loop that reads the lines of each file
for(i in 1:length(files)){
  allData[i] <- readLines(files[i])  
}
allData # vector; each element contains content of text files

# load tm, a text mining package for R
library(tm)
# first create a Corpus, basically a data base for text documents
doc.corpus <- Corpus(VectorSource(allData))
inspect(doc.corpus)
inspect(doc.corpus[3])
# next create a basic Term Document Matrix (rows=terms, columns=documents)
doc.tdm <- TermDocumentMatrix(doc.corpus)
inspect(doc.tdm) 
# notice...
# - all words reduced to lower-case
# - punctuation included
# - only words with 3 or more characters included (see wordLengths option in termFreq documentation)
# If we wanted to include words with two or more characters:
# doc.tdm <- TermDocumentMatrix(doc.corpus, control=list(wordLengths = c(2, Inf)))


# apply transformations to corpus
# make all lower-case in advance; helps with stop words
doc.corpus <- tm_map(doc.corpus, tolower) 
inspect(doc.corpus)
# remove lower-case stop words
# see stop words: stopwords("english")
doc.corpus <- tm_map(doc.corpus, removeWords, stopwords("english")) 
inspect(doc.corpus)
# remove punctuation
doc.corpus <- tm_map(doc.corpus, removePunctuation) 
inspect(doc.corpus)
# remove numbers
doc.corpus <- tm_map(doc.corpus, removeNumbers) 
inspect(doc.corpus)

# if you want to add stopwords:
# tm_map(doc.corpus, removeWords, c(stopwords("english"),"custom","words")) 

doc.tdm <- TermDocumentMatrix(doc.corpus)
inspect(doc.tdm)
# notice...
# - all words reduced to lower case
# - punctuation and numbers gone
# - stopwords gone
# - no distinction between Iron in "Iron Mainden" and the iron for pressing clothes
# - weighting is simple term frequency

# Important to know how Corpora and TDMs are created. They will often be huge and not
# easily checked by eye.

# Weight a term-document matrix by term frequency - inverse document frequency (TF-IDF)
# Idea: words with high term frequency should receive high weight unless they also have
# high document frequency

inspect(weightTfIdf(doc.tdm))


meta(doc.corpus[[1]], tag="Description")  <- "some text"
DublinCore(doc.corpus)


# classification of tweets (JSON) -----------------------------------------

# STEP 1: read in twitter JSON files

# set directory with tweets
setwd("~/workshops/Text Mining/data/json_files/")

# load RJSONIO package which allows us to read in JSON files using fromJSON() function
library(RJSONIO)
files <- dir() # vector of all json file names
files
length(files)
# What does the fromJSON() do?
files[1]
fromJSON(files[1])

# "apply" the fromJSON() function to all JSON files and create a single list object
tdata <- lapply(files, fromJSON) 
tdata[[1]] # see first tweet

# STEP 2: prep twitter data

# extract tweets
tdata[[1]]$text # first tweet
# load all tweets into a vector
tweets <- sapply(tdata,function(x) x$text)

# see random sample of 10 tweets
set.seed(1) # so we all seee the same tweets
sample(tweets,10)

# create variable measuring days after resignation;
# 2012-06-08: Dragas and Kington ask for Sullivan's resignation
# extract date of tweet
dates <- as.Date(sapply(tdata,function(x) x$created_at), format="%a %b %d %H:%M:%S %z %Y")
# days after 2012-06-08
daysAfter <- as.numeric(dates - as.Date("2012-06-08"))

# remove some objects to free up some memory
rm(tdata, dates, files)

# STEP 3: create DocumentTermMatrix

library(tm)
# create corpus
corpus <- Corpus(VectorSource(tweets)) # this takes a few moments

# clean up corpus
corpus <- tm_map(corpus, tolower)
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, removeWords, stopwords('english'))
corpus <- tm_map(corpus, removePunctuation) 
corpus <- tm_map(corpus, removeNumbers) 

# create DTM; only keep terms that appear in at least 5 tweets
dtm <- DocumentTermMatrix(corpus, control = list(bounds = list(global = c(5,Inf))))
inspect(dtm[1:5,1:5])
colnames(dtm) # see selected words

# convert document-term matrix (DTM) to matrix form;
# need this for analysis
X <- as.matrix(dtm)
X[1:5,1:5]

# remove some objects to free up some memory
rm(corpus, dtm)

# STEP 4: export sample to classify

set.seed(12)
# extract sample (250 tweets) for classification
sample.tweets <- tweets[sample(1:length(tweets),250)] 
# export 100 sample tweets for indicator to be added
setwd("~/workshops/Text Mining/data/")
write.csv(sample.tweets,"sample.tweets.csv", row.names=FALSE)
# open file in Excel and add indicator (1 = relevant, 0 = not relevant)
# Done prior to workshop

# STEP 5: read in classified tweets and combine with DTM matrix

# read data with indicators (1= relevant, 0=not relevant)
# these data sampled with set.seed(12)
stwt <- read.csv("sample.tweets.csv", header=TRUE)

# need to sample from DTM 
set.seed(12)
sampX <- X[sample(1:length(tweets),250),] 
# need to save what wasn't sampled for classification after model-building
set.seed(12)
restX <- X[-sample(1:length(tweets),250),] 

# add daysAfter to DTM matrices; it will be an additional predictor
set.seed(12)
sampX <- cbind(sampX,daysAfter[sample(1:length(daysAfter),250)]) 
colnames(sampX)[dim(sampX)[2]] <- "days.after"
set.seed(12)
restX <- cbind(restX,daysAfter[-sample(1:length(daysAfter),250)]) 
colnames(restX)[dim(restX)[2]] <- "days.after"

# create a response vector for analysis
Y <- stwt$ind
sum(Y)/length(Y) # percent relevant

# STEP 6: create classification model using the Lasso

# classify using logistic regression (The Lasso)
library(glmnet)
# define a range of lambda values (ie, tuning parameter)
grid <- 10^seq(10,-2, length=100) 
set.seed(1)
# create training and testing sets
train <- sample(1:nrow(sampX), nrow(sampX)/2)
test <- (-train)
y.test <- Y[test]

# fit logistic model via lasso
# alpha=1 is lasso; alpha=0 is ridge regression
# perform cross-validation and find the best lambda (ie, the lowest lambda);
# type.measure="class" gives misclassification error rate
set.seed(1)
cv.out <- cv.glmnet(sampX[train,], factor(Y[train]), alpha=1, family="binomial",  type.measure = "class")
plot(cv.out)
bestlam <- cv.out$lambda.min # best lambda as selected by cross validation


# the lasso model with lambda chosen by cross-validation contains only selected variables
# re-run model with all data using selected lambda
out <- glmnet(sampX, factor(Y), alpha=1, lambda=grid, family="binomial")
lasso.coef <- predict(out, type="coefficients", s=bestlam)
# see selected coefficients
lasso.coef 

# see how it works on training set
sum(ifelse(predict(out, newx = sampX, s = bestlam) > 0, 1, 0) == Y)/250

# STEP 7: classify tweets

# now make predictions for tweets with no classification
predY <- predict(cv.out, newx=restX, s="lambda.min", type="class")

# combine prediction with tweets
classifiedTweets <- cbind(predY, tweets[-sample(1:length(tweets),250)])
# check a few
classifiedTweets[sample(750,1),]


# sentiment analysis (NY Times API) ----------------------------------------
setwd("workshops/Text Mining/data")

# The New York Times API (application programming interfaces) allows you to programmatically 
# access New York Times data for use in your own applications.
# Usage is limited to 5000 requests per day
# request an API key: http://developer.nytimes.com/apps/register (you'll need to Register for NYTimes.com first)

library(rjson)

# Example: request comments on a story using The Community API
# Use a special URL formatted as follows:
# http://api.nytimes.com/svc/community/{version}/comments/url/{match-
# type}[.response-format]?{url=url-to-match}&[offset=int]$&[sort=newest]&api-key={your-API-key}

# And Then Steve Said, 'Let There Be an iPhone'
# By FRED VOGELSTEIN
# Published: October 4, 2013 


# In this example....
# match-type = exact-match
# .response-format = .json
# url = http://www.nytimes.com/2013/10/06/magazine/and-then-steve-said-let-there-be-an-iphone.html

# offset = 25
# sort = recommended

theCall <- "http://api.nytimes.com/svc/community/v2/comments/url/exact-match.json?url=
  http://www.nytimes.com/2013/10/06/magazine/and-then-steve-said-let-there-be-an-iphone.html&
  offset=25&sort=recommended&api-key=85c35331a865573538990d5ddc0c505d:19:28348776"
test <- fromJSON(file=theCall)

# see just the first comment
test$results$comments[[1]]$commentBody

# see all comments
sapply(test$results$comments, function(x)x$commentBody)

# function to help download NY Times comments using API call
# url: link to page article with comments
# offset values: Positive integer, multiple of 25
# sort values: newest | oldest | recommended | replied | editors-selection
nytComments <- function(url,offset,nysort){
  x <- paste("http://api.nytimes.com/svc/community/v2/comments/url/exact-match.json?url=",url,
             "&offset=",offset,
             "&sort=",nysort,
             "&api-key=85c35331a865573538990d5ddc0c505d:19:28348776",sep="")
  y <- fromJSON(file=x)
  z <- sapply(y$results$comments,function(x)x$commentBody)
  z
}

# test it out
nytComments(url="http://www.nytimes.com/2013/10/06/magazine/and-then-steve-said-let-there-be-an-iphone.html",
            offset=25,
            nysort="recommended")

# get 100 comments
# loop through 25, 50, 75, 100
story <- "http://www.nytimes.com/2013/10/06/magazine/and-then-steve-said-let-there-be-an-iphone.html"
comments <- c()
for(i in seq(25,100,25)){ 
comments <- c(comments, nytComments(url=story,
                        offset=as.character(i),
                        nysort="recommended"))
}


# sentiment analysis function

#  implement a very simple algorithm to estimate
#  sentiment, assigning a integer score by subtracting the number 
#  of occurrences of negative words from that of positive words.
#  Courtesy of:
#  https://raw.github.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107/08a269765a6b185d5f3dd522c876043ba9628715/R/sentiment.R

# A list of positive and negative opinion words or sentiment words for English
# (around 6800 words). This list was compiled over many years starting from 
# Hu and Liu, KDD-2004.

# http://www.cs.uic.edu/~liub/FBS/opinion-lexicon-English.rar

poswords <- scan("~/workshops/Text Mining/opinion-lexicon-English/positive-words.txt",
                 what="character", comment.char=";")
negwords <- scan("~/workshops/Text Mining/opinion-lexicon-English/negative-words.txt",
                 what="character", comment.char=";")


sentence <- comments[1]
library(stringr)

score.sentiment <- function(sentence){
      # clean up sentences with R's regex-driven global substitute, gsub():
      sentence <- gsub("<[^>]*>", " ",sentence) # remove html tags
      sentence <- gsub("[[:punct:]]", "", sentence) # Punctuation characters
      sentence <- gsub("[[:cntrl:]]", "", sentence) # Control characters
      sentence <- gsub("\\d+", "", sentence) # digits
      sentence <- tolower(sentence) # convert to lower case
      
      # split into words. str_split is in the stringr package
      word.list <- str_split(sentence, "\\s+")
      # sometimes a list() is one level of hierarchy too much
      words <- unlist(word.list)
      
      # compare our words to the dictionaries of positive & negative terms
      pos.matches <- match(words, poswords)
      neg.matches <- match(words, negwords)
      
      # match() returns the position of the matched term or NA
      # we just want a TRUE/FALSE:
      pos.matches <- !is.na(pos.matches)
      neg.matches <- !is.na(neg.matches)
      
      # and conveniently enough, TRUE/FALSE will be treated as 1/0 by sum():
      score <- sum(pos.matches) - sum(neg.matches)
      score
}

# test the function
sample <- c("You're awesome and I love you",
            "I hate and hate and hate. So angry. Die!",
            "Impressed and amazed: you are peerless in your achievement of unparalleled mediocrity")

# apply function to each sample sentence in sample vector
lapply(sample, score.sentiment) 

# Score the comments and save
scores <- unlist(lapply(comments, score.sentiment))

# summaries
hist(scores)
table(scores)
mean(scores)
median(scores)
range(scores)


# "positive" comments
comments[which(scores==20)]
comments[which(scores==7)]
# "negative" comments
comments[which(scores==-5)]
comments[which(scores==-4)]


# summarize/wordcloud (web scraping) ----------------------------------------
setwd("workshops/Text Mining/data")

# Interested in summarizing reviews of FitBit; want to know more about the negative reviews
# http://www.amazon.com/Fitbit-Wireless-Activity-Tracker-Charcoal/product-reviews/B0095PZHZE/ref=cm_cr_pr_top_link_2?ie=UTF8&pageNumber=1

# STEP 1: develop strategy for scraping data
# read source code page 1 of reviews
test <- readLines("http://www.amazon.com/Fitbit-Wireless-Activity-Tracker-Charcoal/product-reviews/B0095PZHZE/ref=cm_cr_pr_top_link_2?ie=UTF8&pageNumber=1")

# work to get reviews
# get indices of where these phrases occur and count how many; tells us how many reviews on page
length(grep("This review is from:", test))

# text that always appears before review
grep("This review is from:", test)

# test that alwats appears after review
grep( "Help other customers find the most helpful reviews", test)

# get just the first review
test[grep("This review is from:", test)[1]:
       grep( "Help other customers find the most helpful reviews", test)[1]]

# this returns a vector of length 7; the 4th element always contains the review
test[grep("This review is from:", test)[1]:
                 grep( "Help other customers find the most helpful reviews", test)[1]][4]

# therefore this pulls only reviews
test[grep("This review is from:", test)[1] +3]


# work to get stars
# get number of stars for each review
# appears that rating is preceded by following code:
grep("margin-right:5px;\"><span class=\"swSprite s_star_", test)

# pull all lines of code into vector
rating <- test[grep("margin-right:5px;\"><span class=\"swSprite s_star_", test)]
rating
rating <- substr(rating,70,70)
rating


# STEP 2: write formal code to extract reviews
# do not run during workshop; takes too long!

reviews <- rep(NA, 5000) # vector to store reviews
ratings <- rep(NA, 5000) # vector to store ratings
n <- 1 # review counter
i <- 1 # page counter
# Loop until length(grep("This review is from:", test)) == 0
repeat{
  getPage <- readLines(paste("http://www.amazon.com/Fitbit-Wireless-Activity-Tracker-Charcoal/product-reviews/B0095PZHZE/ref=cm_cr_pr_top_link_2?ie=UTF8&pageNumber=",
                             i,sep=""))
  if(length(grep("This review is from:", getPage)) == 0) break else {
    for(j in 1:length(grep("This review is from:", getPage))){
      temp1 <- getPage[grep("margin-right:5px;\"><span class=\"swSprite s_star_", getPage)][j]
      ratings[n] <- substr(temp1,70,70)
      reviews[n]  <- getPage[grep("This review is from:", getPage)[j] + 3]
      n <- n + 1
    }
    i <- i + 1
  }
}
# remove any missing records from each vector
reviews <- reviews[!is.na(reviews)]
ratings <- ratings[!is.na(ratings)]
# make into a data frame  
allReviews <- data.frame(review=reviews, rating=ratings, stringsAsFactors=FALSE)
# remove HTML tags from text of reviews
allReviews$review <- gsub("<[^>]*>", " ",allReviews$review) 

# save for later use
save(allReviews, file="amzReviews.Rda")

# load this for workshop
load("amzReviews.Rda") # collected 16-Jan-2014

# STEP 3: summarize and investigate

# avg customer review
mean(as.numeric(allReviews$rating))
# distribution of ratings
table(allReviews$rating)
barplot(table(allReviews$rating))

# let's look at the 1 star reviews
badReviews <- subset(allReviews, subset= rating=="1")
dim(badReviews)

# make Term-Document matrix of words
library(tm)
revCorp <- Corpus(VectorSource(badReviews$review))
revCorp <- tm_map(revCorp, tolower)
revCorp <- tm_map(revCorp, stripWhitespace)
revCorp <- tm_map(revCorp, removeWords, c(stopwords("english"),"fitbit", "zip"))
revCorp <- tm_map(revCorp, removePunctuation) 
revCorp <- tm_map(revCorp, removeNumbers) 

# wordcloud from corpus 
library("wordcloud")
wordcloud(revCorp,min.freq=15)
wordcloud(revCorp,min.freq=15, random.order=F, rot.per=0) # less artsy

# wordcloud from frequency counts
tdm <- TermDocumentMatrix(revCorp)
m <- as.matrix(tdm)
v <- sort(rowSums(m),decreasing=TRUE)
v[1:20] # top 20 most frequent terms
d <- data.frame(word = names(v),freq=v)
wordcloud(d$word,d$freq, min.freq=15)


library(RColorBrewer)
pal <- brewer.pal(4,"Dark2")
wordcloud(words=d$word,freq=d$freq,min.freq=15,colors=pal)
  
findFreqTerms(tdm, lowfreq=15)
# find associations for the term "battery"
findAssocs(tdm, "battery", 0.5)

# create a dictionary of words;
# we'll use this to create a DTM for just those words
d <- Dictionary(c("battery", "work", "service"))
inspect(DocumentTermMatrix(revCorp, list(dictionary = d)))
# certain documents mention battery a lot

# save the DTM
dtm <- DocumentTermMatrix(revCorp, list(dictionary = d))
# extract the rownames (doc numbers) where battery mentioned more than 2 times
ind <- rownames(subset(as.matrix(dtm), subset= as.matrix(dtm)[,1]>2) )
# see the reviews where battery mentioned more than 2 times
badReviews$review[as.numeric(ind)]

# cluster analysis: do they fall naturall into groups
# k-means clustering - have to specify number of groups
azRev.sd <- scale(dtm)
set.seed(9)
km.out <- kmeans(azRev.sd, 6, nstart=50)
km.clusters <- km.out$cluster  
table(km.clusters)

# add class membership to data frame
badReviews[,"km.clusters"] <- km.clusters

# how are they similar within clusters?
badReviews[badReviews$km.clusters==1,"review"]
badReviews[badReviews$km.clusters==5,"review"]



#############################################################################
#############################################################################
#############################################################################
#############################################################################