import asyncio
from avanza import Avanza, ChannelType
from time import time, ctime

def callback(data):
    # Do something with the quotes data here
    time = ctime(data['data']['dealTime']/1000)
    price = data['data']['price']
    #sell = data['data']['sellPrice']
    vol = data['data']['volume']
    
    string = f'time: {time}, price: {price}, vol: {vol}'

    print(data)

async def subscribe_to_channel(avanza: Avanza):
    await avanza.subscribe_to_id(
        ChannelType.TRADES,
        "549768",
        callback
    )

def main():
    avanza = Avanza({
    'username': 'HjartarsonErik',
    'password': 'uzLfiSqA',
    'totpSecret': 'HRN33P7MFHRHY6VH5HQHR54RGEH6NODQ'
    })

    asyncio.get_event_loop().run_until_complete(
        subscribe_to_channel(avanza)
    )
    asyncio.get_event_loop().run_forever()

if __name__ == "__main__":
    main()