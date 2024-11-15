"""Utility functions for the eBay Christmas Deals DAG"""

from datetime import datetime

def create_finding_api_payload(keywords, category_id, page_number, entries_per_page):
    """Create a properly formatted payload for the eBay Finding API"""
    return {
        'findItemsAdvancedRequest': {
            'xmlns': 'http://www.ebay.com/marketplace/search/v1/services',
            'keywords': keywords,
            'categoryId': category_id,
            'paginationInput': {
                'pageNumber': str(page_number),
                'entriesPerPage': str(entries_per_page)
            },
            'itemFilter': [
                {'name': 'ListingType', 'value': 'FixedPrice'},
                {'name': 'HideDuplicateItems', 'value': 'true'}
            ],
            'outputSelector': ['SellerInfo', 'StoreInfo', 'PictureURLLarge']
        }
    }

def process_item_data(item, category_id, marketplace):
    """Process an individual item from eBay API response"""
    selling_status = item.get('sellingStatus', {})
    current_price = selling_status.get('currentPrice', {})
    
    return {
        "Title": item.get('title', "N/A"),
        "Item ID": item.get('itemId', "N/A"),
        "Price": float(current_price.get('value', 0)),
        "Currency": current_price.get('_currencyId', "N/A"),
        "Image URL": item.get('galleryURL', "N/A"),
        "View Item URL": item.get('viewItemURL', "N/A"),
        "Category ID": category_id,
        "Marketplace": marketplace,
        "Timestamp": datetime.now().isoformat(),
        "Seller": item.get('sellerInfo', {}).get('sellerUserName', "N/A"),
        "Condition": item.get('condition', {}).get('conditionDisplayName', "N/A"),
        "Location": item.get('location', "N/A"),
        "Listing Type": item.get('listingInfo', {}).get('listingType', "N/A"),
        "Best Offer": item.get('listingInfo', {}).get('bestOfferEnabled', "N/A"),
        "Buy It Now": item.get('listingInfo', {}).get('buyItNowAvailable', "N/A"),
        "Shipping Type": item.get('shippingInfo', {}).get('shippingType', "N/A"),
        "Shipping Cost": item.get('shippingInfo', {}).get('shippingServiceCost', {}).get('value', "N/A"),
        "Returns Accepted": item.get('returnsAccepted', "N/A"),
        "Top Rated": item.get('topRatedListing', False)
    }