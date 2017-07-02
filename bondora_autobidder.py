# -*- coding: utf-8 -*-
from bondora_rf_default_predictor import BondoraPredictor
from requests_oauthlib import OAuth2Session
import os
import time
import logging
import json

with open('config.json', 'r') as f:
    config = json.load(f)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler('bondora.log')
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
predictor = BondoraPredictor()

client_id = config['ClientId']
client_secret = config['ClientSecret']
authorization_base_url = config['AuthorizationBaseUrl']
token_url = config['TokenUrl']
scope = config['Scope']

if('Token' in config):
    print("Using token from configuration file")
    bondora = OAuth2Session(client_id=client_id, token=config['Token'])
else:
    bondora = OAuth2Session(client_id=client_id, auto_refresh_url=token_url, scope=scope)
    authorization_url, state = bondora.authorization_url(authorization_base_url)
    print('Please go to this URL and authorize access:')
    print(authorization_url)
    print("")
    print("Enter the full callback URL:")
    authorization_response = input('')
    token = bondora.fetch_token(token_url, authorization_response=authorization_response, client_id=client_id, client_secret=client_secret)
    print("Got new access token:")
    print(token)


logger.info("Initiating Bondora autobidder")
while(True):

    available_auctions = []
    interesting_auctions = []
    already_bidded_auctions = []
    my_bids = []

    try:
        available_auctions = bondora.get('https://api.bondora.com/api/v1/auctions').json()['Payload']
        my_bids = bondora.get('https://api.bondora.com/api/v1/bids').json()['Payload']
    except:
        logger.exception("Failed getting data from Bondora")


    for a in available_auctions:
        interesting = True
        for b in my_bids:
            if(b['AuctionId'] == a['AuctionId']):
                already_bidded_auctions.append(a)
                interesting = False
                break

        if (interesting):
            a['Rating_V2'] = predictor.transformValue('Rating_V2',[a['Rating']])[0]
            a['Prediction'] = predictor.predict(a)

            if(a['Prediction']):
                interesting_auctions.append(a)

    if len(available_auctions) > 0:
        logger.info("Auctions: {}, invested: {}, interesting: {}".format(len(available_auctions), len(already_bidded_auctions), len(interesting_auctions)))
        for a in available_auctions:
            logger.info("Auction: {}".format(a))     

        for a in interesting_auctions:
            try:
                balance = bondora.get('https://api.bondora.com/api/v1/account/balance').json()['Payload']
                logger.info(balance)
            except:
                logger.exception("Failed getting balance data from Bondora")
                break

            if(balance['TotalAvailable'] > 5):
                logger.info("Bidding for auction: {}".format(a['AuctionId']))

                json_data = {
                      "Bids": [
                        {
                          "AuctionId": a['AuctionId'],
                          "Amount": 5.0,
                          "MinAmount": 1.0
                        }
                      ]
                    }

                bid_result = bondora.post('https://api.bondora.com/api/v1/bid', json=json_data).json()
                logger.info("Bidding success={}".format(bid_result['Success']))
            else:
                logger.info("Insufficient funds for bidding ({})".format(balance['TotalAvailable']))


    time.sleep(10)
